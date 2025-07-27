"""
Some functions which help to build the index of src paths/urls for a dataset.
"""
import os
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import polars as pl
from loguru import logger
from tqdm import tqdm

from sds.utils.download import CloudDownloader
from sds.structs import DataSampleType, DATA_TYPE_TO_EXT
import sds.utils.distributed as dist_utils
import sds.utils.os_utils as os_utils
import sds.utils.data_utils as data_utils

#---------------------------------------------------------------------------
# Data structures and constants for the index.

class IndexType(Enum):
    INTER_NODE = 'INTER_NODE'
    INTRA_NODE = 'INTRA_NODE'

    # You can add this for a friendlier string representation
    def __str__(self):
        return self.value

INDEX_FILE_NAME = 'index.parquet' # The name of the index file to be saved on disk.
RAW_INDEX_FILES_DIR = 'raw_index_files' # The directory where raw index files will be saved.

@dataclass(frozen=True)
class IndexMetaData:
    num_samples: int
    path: str
    index_type: IndexType # This affects how we slice the data across nodes and ranks.

#---------------------------------------------------------------------------

def build_index(src: str, dst_dir: str, data_type: DataSampleType, **kwargs) -> IndexMetaData:
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
        return build_index_from_many_index_files(src, dst_dir, **kwargs)
    elif any(src.endswith(ext) for ext in ['.csv', '.json', '.parquet']): # TODO: process parquet data more intelligently via slicing.
        return build_index_from_index_file(src, dst_dir, **kwargs)
    else:
        files_list = os_utils.find_files_in_src(src)
        assert files_list, f"No files found in the source {src} for data type {data_type}."
        return build_index_from_files_list(files_list, data_type=data_type, dst_dir=dst_dir, **kwargs)


def build_index_from_many_index_files(src: str, dst_dir: str, shuffle_seed: int, max_size: int=None, cols_to_keep: list[str] | None=None) -> IndexMetaData:
    """
    This function builds an index from either `split_file_paths.txt` list or wildcard path (e.g., 's3://bucket/path/*.csv').
    It's an intra-node index, meaning that each node will process its own subset of data.
    """
    index_type = IndexType.INTRA_NODE
    # We are processing a list of CSV/JSON/PARQUET files (possibly stored in S3).
    if src.endswith('/split_file_paths.txt'):
        # That's a special case: we receive a file containing a list of index paths, one per line.
        # Let's load it, read and distribute the data across ranks.
        dst = os.path.join(dst_dir, RAW_INDEX_FILES_DIR, 'split_file_paths.txt')
        CloudDownloader.get(src).direct_download(remote=src, local=dst)
        with open(dst, 'r') as f:
            index_files_list = [line.strip() for line in f if line.strip()]
        src_exts = {os_utils.file_ext(f).lower() for f in index_files_list}
        assert len(src_exts) == 1, f"Expected all files to have the same extension, but found: {src_exts}. This is an SDS bug, and we have a problem with data processing."
        src_ext = src_exts.pop()  # Get the single extension from the set.
    else:
        # Index files are passed as a wildcard path (e.g., 's3://bucket/path/*.csv'). Let's find them all.
        src_ext = os_utils.file_ext(src).lower()
        index_files_list = os_utils.find_files_in_src(src.replace(f'*{src_ext}', ''), exts={src_ext}) # Remove the wildcard from the src path.

    assert src_ext in ['.csv', '.json', '.parquet'], f"Expected the index files to be in CSV, JSON or PARQUET format, but found: {src_ext}. This is an SDS bug, and we have a problem with data processing."

    # Once we have the list of files, we need to distribute them across nodes.
    # We distribute the data across nodes on a per-file basis instead of per-sample basis. This mainly affects shuffling.
    node_rank = dist_utils.get_node_rank()
    num_files_per_node = len(index_files_list) // dist_utils.get_num_nodes()
    assert num_files_per_node > 0, f"Not enough files to distribute across nodes. Found {len(index_files_list)} files, but expected at least {dist_utils.get_num_nodes()} files per node."
    np.random.RandomState(shuffle_seed).shuffle(index_files_list)  # Shuffle the files list for randomness.
    cur_node_files_list = index_files_list[node_rank * num_files_per_node:(node_rank + 1) * num_files_per_node]

    # Now, we need to download them in parallel and save as a unified parquet file.
    dst_files_list = [os.path.join(dst_dir, RAW_INDEX_FILES_DIR, os.path.basename(f)) for f in cur_node_files_list]
    os_utils.parallel_download(cur_node_files_list, dst_files_list, skip_if_exists=True, num_workers=16, verbose=True)

    # Now, we can concatenate the data from all the files into a single DataFrame.
    reader = {'.csv': pd.read_csv, '.json': pd.read_json, '.parquet': lambda *args, **kwargs: pq.read_table(*args, **kwargs).to_pandas()}[src_ext]
    index_list_raw_size = sum([os_utils.get_file_size(f) for f in dst_files_list])
    df = pd.concat((reader(f) for f in tqdm(dst_files_list, desc=f"Loading in memory {len(dst_files_list)} <index>{src_ext} files. Raw size: {index_list_raw_size:,} bytes.")), ignore_index=True)
    df = maybe_shuffle_df(df, shuffle_seed)
    df = maybe_slice_df(df, max_size, index_type, cols_to_keep=cols_to_keep)
    index_dst = os.path.join(dst_dir, INDEX_FILE_NAME)
    logger.debug(f"Saving the index to {index_dst} with {len(df):,} samples...")
    data_utils.save_polars_parquet(df, index_dst)
    index_meta = IndexMetaData(len(df), index_dst, index_type)  # Placeholder for the actual number of samples.

    return index_meta


def build_index_from_index_file(src: str, dst_dir: str, shuffle_seed: int=None, max_size: int=None, cols_to_keep: list[str] | None=None) -> IndexMetaData:
    index_type = IndexType.INTER_NODE
    # We have just a single index file which contains all the data samples metadata.
    # First, download the file to the destination directory.
    dst = os.path.join(dst_dir, RAW_INDEX_FILES_DIR, os.path.basename(src))
    logger.debug(f"Downloading the index file from {src} to {dst}...")
    assert os_utils.download_file(src, dst, skip_if_exists=True), f"Failed to download the index file from {src} to {dst}."
    assert os_utils.is_non_empty_file(dst), f"Failed to download the index file from {src} to {dst}."

    # Reading the file.
    src_ext = os_utils.file_ext(src).lower()
    reader = {'.csv': pd.read_csv, '.json': pd.read_json, '.parquet': pq.read_table}[src_ext]
    logger.debug(f"Reading the index file {dst} into memory... Size: {os_utils.get_file_size(dst):,} bytes.")
    df = reader(dst)
    if isinstance(df, pa.Table): # Convert to pandas DataFrame if it's a PyArrow Table.
        logger.debug(f"Converting the index file from PyArrow Table to pandas DataFrame...")
        df = df.to_pandas()
    df = maybe_shuffle_df(df, shuffle_seed)
    df = maybe_slice_df(df, max_size, index_type, cols_to_keep=cols_to_keep)
    assert isinstance(df, pd.DataFrame), f"Expected a DataFrame, got {type(df)} from {src}."

    # Now, we can save it as a parquet file for easier slicing.
    index_dst = os.path.join(dst_dir, INDEX_FILE_NAME)
    logger.debug(f"Saving the index to {index_dst} with {len(df):,} samples...")
    data_utils.save_polars_parquet(df, index_dst)

    return IndexMetaData(len(df), index_dst, index_type)


def build_index_from_files_list(files_list: list[str], dst_dir: str, data_type: DataSampleType, shuffle_seed: int=None, max_size: int=None, cols_to_keep: list[str] | None=None) -> IndexMetaData:
    index_type = IndexType.INTER_NODE
    main_files = [f for f in files_list if os_utils.file_ext(f).lower() in DATA_TYPE_TO_EXT[data_type]]
    assert len(main_files) > 0, f"Didnt find any {data_type} files (used extensions: {DATA_TYPE_TO_EXT[data_type]})."
    main_file_keys = list(set([os_utils.file_key(f) for f in main_files]))
    assert len(main_file_keys) > 0, f"Didnt find any {data_type} files (used extensions: {DATA_TYPE_TO_EXT[data_type]})."

    # Now, once we have the main files, we can build the columns.
    # For this, we first want to group the files by their keys using the existing keys as prefixes.
    # We must be careful since some files are named {key}.{ext1}.{ext2}.{...}.json
    data = {k: {} for k in main_file_keys}  # Initialize a dict with keys as the base names of the files.
    for file in files_list:
        key = os_utils.file_key(file) # Get the key (base name without extension) of the file.
        full_ext = os_utils.file_full_ext(file).lower() # Get the full extension (e.g., .jpg, .txt, etc.)
        if key not in data:
            continue # Skipping a file since it's some random file which is not matched with the main keys.
        assert full_ext not in data[key], f"Duplicate key found: {key} with extension {full_ext}. This is an SDS bug, and we have a problem with data processing."
        data[key][full_ext[1:]] = file # Store the file path under the key and extension.

    # Convert the dict to a DataFrame
    INDEX_COL_NAME = 'index'
    df = pd.DataFrame.from_dict(data, orient='index').reset_index(names=INDEX_COL_NAME).sort_values(by=INDEX_COL_NAME)
    df = maybe_shuffle_df(df, shuffle_seed)
    df = maybe_slice_df(df, max_size, index_type, cols_to_keep=cols_to_keep)
    index_dst = os.path.join(dst_dir, INDEX_FILE_NAME)
    data_utils.save_polars_parquet(df, index_dst)
    index_meta = IndexMetaData(len(df), index_dst, index_type)

    return index_meta

#---------------------------------------------------------------------------
# Loading functions for an already created index.

def load_index_slice(index_meta: IndexMetaData, rank: int, num_ranks: int, num_nodes: int) -> pd.DataFrame:
    assert index_meta.path.endswith('.parquet'), f"Index file must be a parquet file. Found: {index_meta.path}"
    start_time = time.time()
    start_idx, num_samples_per_rank = compute_index_slice(index_meta, rank, num_ranks, num_nodes)
    logger.debug(f"Loading index slice for rank {rank} (start_idx={start_idx}, num_samples_per_rank={num_samples_per_rank}) from {index_meta.path}")
    index_slice = data_utils.read_parquet_slice(index_meta.path, start_idx, num_samples_per_rank)
    logger.debug(f"Loaded index slice for rank {rank} with {len(index_slice):,} samples. Time taken: {time.time() - start_time:.2f} seconds.")
    return index_slice


def load_index_row(index_meta: IndexMetaData, idx: int) -> pd.DataFrame:
    """
    Loading just a single row from the index file.
    """
    assert index_meta.path.endswith('.parquet'), f"Index file must be a parquet file. Found: {index_meta.path}"
    row_df = pl.scan_parquet(index_meta.path).slice(offset=idx, length=1).collect().to_pandas()
    return row_df

def compute_index_slice(index_meta: IndexMetaData, rank: int, num_ranks: int, num_nodes: int) -> tuple[int, int]:
    if index_meta.index_type == IndexType.INTRA_NODE:
        # Each node has its own slicing.
        num_ranks_per_node = num_ranks // num_nodes
        num_samples_per_rank = index_meta.num_samples // num_ranks_per_node
        local_rank = rank % num_ranks_per_node
        start_idx = local_rank * num_samples_per_rank
    elif index_meta.index_type == IndexType.INTER_NODE:
        num_samples_per_rank = index_meta.num_samples // num_ranks
        start_idx = rank * num_samples_per_rank
    else:
        raise ValueError(f"Unknown index type: {index_meta}")

    return start_idx, num_samples_per_rank

#---------------------------------------------------------------------------
# Miscellaneous transforms.

def maybe_shuffle_df(df: pd.DataFrame, shuffle_seed: int | None) -> pd.DataFrame:
    """
    Shuffle the DataFrame if a shuffle seed is provided.
    """
    if shuffle_seed is not None:
        # TODO: we need a more memory-efficient way to shuffle the DataFrame.
        df = df.sample(frac=1, replace=False, random_state=shuffle_seed)
    return df

def maybe_slice_df(df: pd.DataFrame, max_size: int | None, index_type, cols_to_keep: list[str] | None=None) -> pd.DataFrame:
    """
    Slice the DataFrame to a maximum size if specified.
    """
    if max_size is None:
        return df
    max_size = (max_size // dist_utils.get_num_nodes()) if index_type == IndexType.INTRA_NODE else max_size
    assert max_size > 0, f"Max size must be greater than 0, got {max_size}."
    df = df.head(max_size) if len(df) > max_size else df
    df = df[cols_to_keep] if cols_to_keep is not None else df # Keep only the specified columns if provided.
    return df

#---------------------------------------------------------------------------
