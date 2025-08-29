"""
Some functions which help to build the index of src paths/urls for a dataset.
"""
import os
import time
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from beartype import beartype
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
    lazy: bool = False # Whether the index is lazy-loaded or not. If True, the index will be loaded only when needed.

#---------------------------------------------------------------------------

def build_index(src: str, dst_dir: str, data_type: DataSampleType, max_index_files_to_use: int | None=None, lazy: bool=False, **kwargs) -> IndexMetaData:
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
        return build_index_from_many_index_files(src, dst_dir, max_index_files_to_use=max_index_files_to_use, lazy=lazy, **kwargs)
    elif any(src.endswith(ext) for ext in ['.csv', '.json', '.parquet']): # TODO: process parquet data more intelligently via slicing.
        assert max_index_files_to_use is None, f"max_index_files_to_use is not supported for folder datasets. Got {max_index_files_to_use}."
        return build_index_from_index_file(src, dst_dir, lazy=lazy, **kwargs)
    else:
        files_list = os_utils.find_files_in_src(src)
        assert files_list, f"No files found in the source {src} for data type {data_type}."
        assert not lazy, f"lazy is not supported for folder datasets. Got {lazy}."
        assert max_index_files_to_use is None, f"max_index_files_to_use is not supported for folder datasets. Got {max_index_files_to_use}."
        return build_index_from_files_list(files_list, data_type=data_type, dst_dir=dst_dir, **kwargs)


def build_index_from_many_index_files(src: str, dst_dir: str, shuffle_seed: int, max_size: int=None, lazy: bool=False, max_index_files_to_use: int | None=None, **slicing_kwargs) -> IndexMetaData:
    """
    This function builds an index from either `split_file_paths.txt` list or wildcard path (e.g., 's3://bucket/path/*.csv').
    It's an intra-node index, meaning that each node will process its own subset of data.
    """
    # We are processing a list of CSV/JSON/PARQUET files (possibly stored in S3).
    if src.endswith('/split_file_paths.txt'):
        # That's a special case: we receive a file containing a list of index paths, one per line.
        # Let's load it, read and distribute the data across ranks.
        assert not lazy, f"lazy is not supported for split_file_paths.txt. Got {lazy}."
        dst = os.path.join(dst_dir, RAW_INDEX_FILES_DIR, 'split_file_paths.txt')
        CloudDownloader.get(src).direct_download(remote=src, local=dst)
        with open(dst, 'r') as f:
            index_files_list = sorted([line.strip() for line in f if line.strip()])
    else:
        # Index files are passed as a wildcard path (e.g., 's3://bucket/path/*.csv'). Let's find them all.
        src_ext = os_utils.file_ext(src).lower()
        assert src_ext == '.parquet' or not lazy, f"lazy is only supported for parquet files, got: {src_ext}."
        if lazy:
            num_samples_total = pl.scan_parquet(src, extra_columns='ignore').select(pl.count()).collect().item()
            num_samples_total = min(num_samples_total, max_size) if max_size is not None else num_samples_total
            return IndexMetaData(num_samples=num_samples_total, path=src, index_type=IndexType.INTER_NODE, lazy=True)
        else:
            index_files_list = sorted(os_utils.find_files_in_src(src.replace(f'*{src_ext}', ''), exts={src_ext})) # Remove the wildcard from the src path.

    assert len(index_files_list) > 0, f"No index files found in the source {src}. Please provide a valid source path or URL with index files."
    assert max_index_files_to_use != 0, f"max_index_files_to_use must be greater than 0 or be None, got {max_index_files_to_use}."
    index_files_list = index_files_list[:max_index_files_to_use] if max_index_files_to_use is not None else index_files_list

    # Once we have the list of files, we need to distribute them across nodes.
    # We distribute the data across nodes on a per-sample basis, but there is a catch: some nodes might have more samples than others.
    # So we first need to get the index file sizes, and then re-distribute them.
    node_rank = dist_utils.get_node_rank()
    num_files_per_node = len(index_files_list) // dist_utils.get_num_nodes()
    assert num_files_per_node > 0, f"Not enough files to distribute across nodes. Found {len(index_files_list)} files, but expected at least {dist_utils.get_num_nodes()} files per node."
    np.random.RandomState(shuffle_seed).shuffle(index_files_list)  # Shuffle the files list for randomness.
    cur_node_files_list: list[str] = index_files_list[max(node_rank * num_files_per_node - 1, 0):(node_rank + 1) * num_files_per_node + 1] # Downloading with a slight overlap to ensure all the nodes have all the files cumulatively.
    cur_dfs: dict[str, pd.DataFrame] = load_index_files(cur_node_files_list, dst_dir, already_loaded={})
    sample_counts_local: dict[str, int] = {f: len(df) for f, df in cur_dfs.items()}
    sample_counts_all: dict[str, int] = dist_utils.merge_dicts_across_local_masters(sample_counts_local)

    # Now, we can decide how to slice the data across nodes.
    # We will slice the data based on the number of samples in each file.
    slice_bounds: dict[str, tuple[int, int]] = compute_slicing_bounds(sample_counts_all, num_nodes=dist_utils.get_num_nodes())[node_rank]
    node_files_list = [index_files_list[i] for i, bounds in enumerate(slice_bounds.values()) if bounds[1] > bounds[0]] # Filter files that are part of the current node's slice.
    dfs = load_index_files(node_files_list, dst_dir, already_loaded=cur_dfs)
    assert len(dfs) > 0, f"Failed to load any index files for the current node. Files: {node_files_list}. This is likely an SDS bug, and we have a problem with data processing."

    index_type = IndexType.INTRA_NODE
    df = pd.concat([dfs[f].iloc[slice_bounds[f][0]:slice_bounds[f][1]] for f in dfs], ignore_index=True)
    df = maybe_shuffle_df(df, shuffle_seed)
    df = maybe_slice_df(df, max_size, index_type, **slicing_kwargs)
    index_dst = os.path.join(dst_dir, INDEX_FILE_NAME)
    logger.debug(f"[Node {node_rank}] Saving the index to {index_dst} with {len(df):,} samples...")
    data_utils.save_polars_parquet(df, index_dst)
    index_meta = IndexMetaData(len(df), index_dst, index_type)  # Placeholder for the actual number of samples.

    return index_meta


def build_index_from_index_file(src: str, dst_dir: str, shuffle_seed: int=None, max_size: int=None, lazy: bool=False, **slicing_kwargs) -> IndexMetaData:
    index_type = IndexType.INTER_NODE

    if lazy:
        assert src.endswith('.parquet'), f"lazy is only supported for parquet files, got: {src}."
        num_samples_total = pl.scan_parquet(src, extra_columns='ignore').select(pl.count()).collect().item()
        num_samples_total = min(num_samples_total, max_size) if max_size is not None else num_samples_total
        return IndexMetaData(num_samples=num_samples_total, path=src, index_type=index_type, lazy=True)

    # We have just a single index file which contains all the data samples metadata.
    # First, download the file to the destination directory.
    dst = os.path.join(dst_dir, RAW_INDEX_FILES_DIR, os_utils.path_key(src))  # Use the path key to avoid conflicts.
    logger.debug(f"Downloading the index file from {src} to {dst}...")
    assert os_utils.download_file(src, dst, skip_if_exists=True), f"Failed to download the index file from {src} to {dst}."
    assert os_utils.is_non_empty_file(dst), f"Failed to download the index file from {src} to {dst}."

    logger.debug(f"Reading the index file from {dst} into memory, filtering/slicing and saving as parquet...")
    df = next(iter(load_index_files([src], dst_dir, already_loaded={}).values()))  # Download and load the file into memory as a DataFrame.
    if isinstance(df, pa.Table): # Convert to pandas DataFrame if it's a PyArrow Table.
        logger.debug(f"Converting the index file from PyArrow Table to pandas DataFrame...")
        df = df.to_pandas()
    df = maybe_shuffle_df(df, shuffle_seed)
    df = maybe_slice_df(df, max_size, index_type, **slicing_kwargs)
    assert isinstance(df, pd.DataFrame), f"Expected a DataFrame, got {type(df)} from {src}."

    # Now, we can save it as a parquet file for easier slicing.
    index_dst = os.path.join(dst_dir, INDEX_FILE_NAME)
    logger.debug(f"Saving the index to {index_dst} with {len(df):,} samples...")
    data_utils.save_polars_parquet(df, index_dst)

    return IndexMetaData(len(df), index_dst, index_type)


def build_index_from_files_list(files_list: list[str], dst_dir: str, data_type: DataSampleType, shuffle_seed: int=None, max_size: int=None, **slicing_kwargs) -> IndexMetaData:
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
        assert full_ext not in data[key], f"Duplicate key found: {key} with extension {full_ext}. This is likely an SDS bug, and we have a problem with data processing."
        data[key][full_ext[1:]] = file # Store the file path under the key and extension.

    # Convert the dict to a DataFrame
    INDEX_COL_NAME = 'index'
    df = pd.DataFrame.from_dict(data, orient='index').reset_index(names=INDEX_COL_NAME).sort_values(by=INDEX_COL_NAME)
    df = maybe_shuffle_df(df, shuffle_seed)
    df = maybe_slice_df(df, max_size, index_type, **slicing_kwargs)
    index_dst = os.path.join(dst_dir, INDEX_FILE_NAME)
    data_utils.save_polars_parquet(df, index_dst)
    index_meta = IndexMetaData(len(df), index_dst, index_type)

    return index_meta

#---------------------------------------------------------------------------
# Loading functions for an already created index.

def load_index_partition(index_meta: IndexMetaData, rank: int, num_ranks: int, num_nodes: int) -> pd.DataFrame:
    assert index_meta.path.endswith('.parquet'), f"Index file must be a parquet file. Found: {index_meta.path}"
    start_time = time.time()
    start_idx, end_idx, _step = compute_index_slice(index_meta, rank, num_ranks, num_nodes)
    logger.debug(f"Loading index slice for rank {rank} (start_idx={start_idx}, end_idx={end_idx}) from {index_meta.path}.")
    index_slice = data_utils.read_parquet_slice(index_meta.path, start_idx, end_idx)
    logger.debug(f"Loaded index slice for rank {rank} with {len(index_slice):,} samples. Time taken: {time.time() - start_time:.2f} seconds.")
    return index_slice


def load_index_row(index_meta: IndexMetaData, idx: int) -> pd.DataFrame:
    """
    Loading just a single row from the index file.
    """
    assert index_meta.path.endswith('.parquet'), f"Index file must be a parquet file. Found: {index_meta.path}"
    row_df = pl.scan_parquet(index_meta.path, extra_columns='ignore').slice(offset=idx, length=1).collect().to_pandas()
    return row_df


def compute_index_slice(index_meta: IndexMetaData, rank: int, num_ranks: int, num_nodes: int, interleaved: bool=False) -> tuple[int, int, int]:
    if index_meta.index_type == IndexType.INTRA_NODE:
        # Each node has its own slicing.
        num_ranks_per_node = num_ranks // num_nodes
        num_samples_per_rank = index_meta.num_samples // num_ranks_per_node
        local_rank = rank % num_ranks_per_node
        start_idx = local_rank if interleaved else (local_rank * num_samples_per_rank)
        step = num_ranks_per_node if interleaved else 1
    elif index_meta.index_type == IndexType.INTER_NODE:
        num_samples_per_rank = index_meta.num_samples // num_ranks
        start_idx = rank if interleaved else (rank * num_samples_per_rank)
        step = num_ranks if interleaved else 1
    else:
        raise ValueError(f"Unknown index type: {index_meta}")

    # E.g., start_idx = 0, step = 3, num_samples_per_rank = 5:
    # end_idx = 0 + 5 * 3 = 15, so the indicies are [0, 3, 6, 9, 12].
    end_idx = min(start_idx + num_samples_per_rank * step, index_meta.num_samples)

    return start_idx, end_idx, step

#---------------------------------------------------------------------------
# Miscellaneous transforms.

@beartype
def maybe_shuffle_df(df: pd.DataFrame, shuffle_seed: int | None) -> pd.DataFrame:
    """
    Shuffle the DataFrame if a shuffle seed is provided.
    """
    if shuffle_seed is not None:
        # TODO: we need a more memory-efficient way to shuffle the DataFrame.
        df = df.sample(frac=1, replace=False, random_state=shuffle_seed)
    return df


@beartype
def maybe_slice_df(df: pd.DataFrame, max_size: int | None, index_type, cols_to_keep: list[str] | None=None, sql_query: str | None=None) -> pd.DataFrame:
    """
    Slice the DataFrame to a maximum size if specified.
    """
    df = data_utils.maybe_run_sql_query_on_dataframe(df, sql_query)
    if max_size is None:
        return df
    max_size = (max_size // dist_utils.get_num_nodes()) if index_type == IndexType.INTRA_NODE else max_size
    assert max_size > 0, f"Max size must be greater than 0, got {max_size}."
    df = df.head(max_size) if len(df) > max_size else df
    df = df[cols_to_keep] if cols_to_keep is not None else df # Keep only the specified columns if provided.
    return df

@beartype
def compute_slicing_bounds(sample_counts: dict[str, int], num_nodes: int) -> list[dict[str, tuple[int, int]]]:
    """
    Compute the slicing bounds for each sub-index based on the sub-index' amount of samples.
    This is used to slice the data correctly across multiple nodes.
    I.e., given {index1: 10, index2: 10, index3: 4}, we redistribute the samples as:
        - num_nodes=2, node_rank=0 -> {index1: (0, 10), index2: (0, 2), index3: (0, 0)}
        - num_nodes=2, node_rank=1 -> {index1: (0, 0), index2: (2, 10), index3: (0, 4)}
    """
    total_samples = sum(sample_counts.values())
    sorted_keys = sorted(sample_counts.keys())

    if total_samples == 0:
        empty_bounds = {key: (0, 0) for key in sorted_keys}
        return [empty_bounds] * num_nodes

    all_node_bounds = []
    for node_rank in range(num_nodes):
        base_samples_per_node = total_samples // num_nodes
        remainder = total_samples % num_nodes
        global_node_start = node_rank * base_samples_per_node + min(node_rank, remainder)
        global_node_end = global_node_start + base_samples_per_node + (1 if node_rank < remainder else 0)

        node_slice_bounds = {}
        cumulative_offset = 0
        for key in sorted_keys:
            count = sample_counts[key]
            global_key_start = cumulative_offset
            global_key_end = cumulative_offset + count
            intersection_start = max(global_node_start, global_key_start)
            intersection_end = min(global_node_end, global_key_end)

            if intersection_start < intersection_end:
                local_start = intersection_start - global_key_start
                local_end = intersection_end - global_key_start
                node_slice_bounds[key] = (local_start, local_end)
            else:
                node_slice_bounds[key] = (0, 0)
            cumulative_offset += count
        all_node_bounds.append(node_slice_bounds)
    return all_node_bounds


@beartype
def load_index_files(index_files_list: list[str], dst_dir: str, already_loaded: dict[str, pd.DataFrame], max_workers: int = 32, **download_kwargs) -> dict[str, pd.DataFrame]:
    """
    Downloads and loads index files in parallel into memory in a concise way.
    """
    # 1. Download files (this part is already concise)
    dst_files_list = [os.path.join(dst_dir, RAW_INDEX_FILES_DIR, os_utils.path_key(f)) for f in index_files_list]
    os_utils.parallel_download(index_files_list, dst_files_list, num_workers=max_workers, verbose=True, **download_kwargs)

    # 2. Determine which files actually need to be loaded
    files_to_load = [f for f in dst_files_list if f not in already_loaded]
    if not files_to_load:
        return already_loaded

    # 3. Define readers and a nested loading function to use with `executor.map`
    readers = {'.csv': pd.read_csv, '.json': pd.read_json, '.parquet': lambda f: pq.read_table(f).to_pandas()}

    def load_file_tuple(filepath: str) -> tuple[str, pd.DataFrame | None]:
        """Loads a file; returns (path, DataFrame) on success or (path, None) on failure."""
        try:
            return filepath, readers[os_utils.file_ext(filepath)](filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return filepath, None

    # 4. Execute in parallel and build result dictionary in one go
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map runs `load_file_tuple` on each file.
        # A dict comprehension consumes the results, filters failures, and builds the dictionary.
        results_iterator = executor.map(load_file_tuple, files_to_load)
        newly_loaded = {
            path: df
            for path, df in tqdm(results_iterator, total=len(files_to_load), desc="Loading files")
            if df is not None
        }

    # 5. Return the merged dictionary of old and new data
    return {**already_loaded, **newly_loaded}

#---------------------------------------------------------------------------
