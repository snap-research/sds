"""
Some functions which help to build the index of src paths/urls for a dataset.
"""
import os
import random
from dataclasses import dataclass
from enum import Enum

import torch
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import polars as pl
from streaming.base.storage import CloudDownloader

from sds.structs import DataSampleType, DATA_TYPE_TO_EXT
import sds.utils.distributed as dist_utils
import sds.utils.os_utils as os_utils
import sds.utils.data_utils as data_utils

#---------------------------------------------------------------------------
# Data structures and constants for the index.

INDEX_TYPE = Enum('IndexType', ['INTER_NODE', 'INTRA_NODE'])
INDEX_FILE_NAME = 'index.parquet' # The name of the index file to be saved on disk.
RAW_INDEX_FILES_DIR = 'raw_index_files' # The directory where raw index files will be saved.

@dataclass(frozen=True)
class IndexMetaData:
    num_samples: int
    path: str
    index_type: INDEX_TYPE # This affects how we slice the data across nodes and ranks.

#---------------------------------------------------------------------------

def build_index(src: str, dst_dir: str, data_type: DataSampleType, shuffle_seed: int) -> IndexMetaData:
    """
    This function builds an index (as pandas dataframe) of the dataset to load.
    Then it saves it (or its chunk) on the local disk as parquet for all the ranks to access.
    Args:
        - src:
            The source path or URL to the dataset index. Can be one of the following:
                - a local directory
                - a remote S3 directory
                - a local index file (csv, json, parquet)
                - a remote index file (e.g., an S3 bucket "file.csv")
        - dst_dir:
            The directory where to save the downloaded files (intermediate and final ones).
        - data_type:
            The type of the main data samples in the dataset (e.g., IMAGE, VIDEO, AUDIO, TEXT, etc.) to build the index for.
            It would be used to select the extension to compute the keys and contrusct the index (for folder datasets).
            We might still have auxiliary data samples of other types, but the main data samples will be of this type.
    Returns the index metadata for each rank to understand where to load from.
    """
    assert dist_utils.is_local_main_process(), "This function should be called only on the main process."
    src_ext = os_utils.file_ext(src).lower()

    if src.endswith('/split_file_paths.txt') or src.endswith(f'*{src_ext}'):
        return build_index_from_many_index_files(src, dst_dir, shuffle_seed)
    elif any(src.endswith(ext) for ext in ['.csv', '.json', '.parquet']): # TODO: process parquet data more intelligently via slicing.
        return build_index_from_index_file(src, dst_dir)
    else:
        files_list = os_utils.find_files_in_src(src, exts=DATA_TYPE_TO_EXT[data_type])
        assert files_list, f"No files found in the source {src} for data type {data_type}."
        return build_index_from_files_list(files_list, data_type=data_type, dst_dir=dst_dir)


def build_index_from_many_index_files(src: str, dst_dir: str, shuffle_seed: int) -> IndexMetaData:
    """
    This function builds an index from either `split_file_paths.txt` list or wildcard path (e.g., 's3://bucket/path/*.csv').
    It's an intra-node index, meaning that each node will process its own subset of data.
    """
    src_ext = os_utils.file_ext(src).lower()
    # We are processing a list of CSV/JSON/PARQUET files (possibly stored in S3).
    if src.endswith('/split_file_paths.txt'):
        # That's a special case: we receive a file containing a list of index paths, one per line.
        # Let's load it, read and distribute the data across ranks.
        dst = os.path.join(dst_dir, 'split_file_paths.txt')
        with dist_utils.leader_first(local=True, skip_non_leaders=True):
            CloudDownloader.get(src).direct_download(remote=src, local=os.path.join(dst_dir, RAW_INDEX_FILES_DIR, 'split_file_paths.txt'))
        with open(dst, 'r') as f:
            files_list = [line.strip() for line in f if line.strip()]
    else:
        # Index files are passed as a wildcard path (e.g., 's3://bucket/path/*.csv'). Let's find them all.
        files_list = os_utils.find_files_in_src(src.replace(f'*{src_ext}', ''), exts={src_ext}) # Remove the wildcard from the src path.

    # Once we have the list of files, we need to distribute them across nodes.
    # We distribute the data across nodes on a per-file basis instead of per-sample basis. This mainly affects shuffling.
    node_rank = dist_utils.get_node_rank()
    num_files_per_node = len(files_list) // dist_utils.get_num_nodes()
    random.RandomState(shuffle_seed).shuffle(files_list)  # Shuffle the files list for randomness.
    cur_node_files_list = files_list[node_rank * num_files_per_node:(node_rank + 1) * num_files_per_node]

    # Now, we need to download them in parallel and save as a unified parquet file.
    dst_files_list = [os.path.join(dst_dir, RAW_INDEX_FILES_DIR, os.path.basename(f)) for f in cur_node_files_list]
    os_utils.parallel_download(cur_node_files_list, dst_files_list, skip_if_exists=True, num_workers=16)

    # Now, we can concatenate the data from all the files into a single DataFrame.
    # downloaded_filesfiles = glob.glob(os.path.join(dst_dir, RAW_INDEX_FILES_DIR, f'*{src_ext}'))
    df = pd.concat((pd.read_csv(f) for f in dst_files_list), ignore_index=True)
    index_dst = os.path.join(dst_dir, INDEX_FILE_NAME)
    df.to_parquet(index_dst, index=False)
    index_meta = IndexMetaData(len(df), index_dst, INDEX_TYPE.INTRA_NODE)  # Placeholder for the actual number of samples.

    return index_meta


def build_index_from_index_file(src: str, dst_dir: str) -> IndexMetaData:
    # We have just a single index file which contains all the data samples metadata.
    # First, download the file to the destination directory.
    dst = os.path.join(dst_dir, RAW_INDEX_FILES_DIR, os.path.basename(src))
    assert os_utils.download_file(src, dst, skip_if_exists=True), f"Failed to download the index file from {src} to {dst}."
    assert os_utils.is_non_empty_file(dst), f"Failed to download the index file from {src} to {dst}."
    print('downloaded index file:', src, dst)

    # Reading the file.
    src_ext = os.path.splitext(src)[1].lower()
    reader = {'.csv': pd.read_csv, '.json': pd.read_json, '.parquet': pq.read_table}[src_ext]
    df = reader(dst)
    if isinstance(df, pa.Table):
        df = df.to_pandas()
    assert isinstance(df, pd.DataFrame), f"Expected a DataFrame, got {type(df)} from {src}."

    # Now, we can save it as a parquet file for easier slicing.
    index_dst = os.path.join(dst_dir, INDEX_FILE_NAME)
    df.to_parquet(index_dst, index=False)

    return IndexMetaData(len(df), index_dst, INDEX_TYPE.INTER_NODE)


def build_index_from_files_list(files_list: list[str], dst_dir: str, data_type: DataSampleType) -> IndexMetaData:
    main_files = [f for f in files_list if os_utils.file_ext(f).lower() in DATA_TYPE_TO_EXT[data_type]]
    assert len(main_files) > 0, f"Didnt find any {data_type} files (used extensions: {DATA_TYPE_TO_EXT[data_type]})."
    main_file_keys = set([os_utils.file_key(f) for f in main_files])

    # Now, once we have the main files, we can build the columns.
    # For this, we first want to group the files by their keys using the existing keys as prefixes.
    # We must be careful since some files are named {key}.{ext1}.{ext2}.{...}.json
    data = {k: {} for k in main_file_keys}  # Initialize a dict with keys as the base names of the files.
    for file in files_list:
        key = os_utils.file_key(file) # Get the key (base name without extension) of the file.
        full_ext = os_utils.file_full_ext(file).lower() # Get the full extension (e.g., .jpg, .txt, etc.)
        assert key in data, f"Key {key} not found in data."
        assert full_ext not in data[key], f"Duplicate key found: {key} with extension {full_ext}. Please ensure unique keys in the dataset."
        data[key][full_ext] = file # Store the file path under the key and extension.

    # Convert the dict to a DataFrame
    INDEX_COL_NAME = 'index'
    df = pd.DataFrame.from_dict(data, orient='index').reset_index(names=INDEX_COL_NAME)
    index_dst = os.path.join(dst_dir, INDEX_FILE_NAME)
    data_utils.save_polars_parquet(df, index_dst)
    index_meta = IndexMetaData(len(df), index_dst, INDEX_TYPE.INTER_NODE)

    return index_meta

#---------------------------------------------------------------------------
# Loading functions for an already created index.

def load_index_slice(index_meta: IndexMetaData, rank: int, num_ranks: int) -> pd.DataFrame:
    assert index_meta.path.endswith('.parquet'), f"Index file must be a parquet file. Found: {index_meta.path}"
    num_samples_per_rank = index_meta.num_samples // num_ranks
    start_idx = rank * num_samples_per_rank
    rank_df = pl.scan_parquet(index_meta.path).slice(offset=start_idx, length=num_samples_per_rank).collect().to_pandas()
    return rank_df

def load_index_row(index_meta: IndexMetaData, idx: int) -> pd.DataFrame:
    """
    Loading just a single row from the index file.
    """
    assert index_meta.path.endswith('.parquet'), f"Index file must be a parquet file. Found: {index_meta.path}"
    row_df = pl.scan_parquet(index_meta.path).slice(offset=idx, length=1).collect().to_pandas()
    return row_df

#---------------------------------------------------------------------------
