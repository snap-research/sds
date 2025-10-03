import os
import time
import math
import types
import hashlib
import base64
import traceback
import faulthandler
from typing import Any, Iterator, Sequence
from collections import deque
from collections.abc import Callable

from beartype import beartype
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset
from loguru import logger

from sds.downloader import ParallelDownloader
from sds.lazy_thread_pool import LazyThreadPool
from sds.structs import DataSampleType, SampleData, SampleTransform
from sds.index import build_index, load_index_partition
import sds.utils.distributed as dist_utils
from sds.utils import os_utils
from sds.utils import data_utils
from sds.utils import misc

#---------------------------------------------------------------------------
# Special fields names.

SAMPLE_DISK_USAGE_FIELD = '__worker_disk_usage__' # The total size of the sample in bytes.
PROCESSED_FIELD = '__is_processed__' # A flag to mark the sample as processed.
SAMPLE_KEY_FIELD = '__sample_key__' # A key for the sample, corresponding to the index column value.
DATA_TYPE_FIELD = '__data_type__' # The data type of the sample, e.g. 'csv', 'json', 'parquet', etc.
SCHEDULE_BATCH_SIZE = 30_000 # The number of samples to schedule for download in one batch on each data worker.
MIN_NUM_PENDING_TASKS_THRESH = {
    DataSampleType.IMAGE: 5_000,
    DataSampleType.VIDEO: 500,
    DataSampleType.AUDIO: 5_000,
    DataSampleType.TEXT: 5_000,
    DataSampleType.IMAGE_LATENT: 5_000,
    DataSampleType.VIDEO_LATENT: 500,
    DataSampleType.AUDIO_LATENT: 5_000,
    DataSampleType.TEXT_LATENT: 5_000,
    None: 5_000,
}

#---------------------------------------------------------------------------

class StreamingDataset(IterableDataset):
    def __init__(self,
        src: str, # a CSV file path, a JSON file path, or a directory path (possibly remote)
        dst: str, # A local directory path where to store the samples
        data_type: DataSampleType | str | None = None, # The type of the data sample (useful when building the index)
        name: str | None=None, # A name for the dataset, used to identify it in the logs and metrics.
        shuffle_seed: int | None=None, # Shuffle seed for the dataset.
        transforms: list[Callable]=None, # A list of data augmentation callbacks to apply to the samples.
        columns_to_download: list[str] | None=None, # The names of the columns to use from the index file.
        index_col_name: str='index', # The name of the column to use as the index column. For folder dataset, must be `index`.
        num_downloading_workers: int=4, # The number of workers to use for downloading the samples in parallel.
        prefetch: int=10, # The number of samples to prefetch in the downloader.
        num_downloading_retries: int=3, # The number of retries to download a sample if it fails.
        none_to_empty_str: bool=True, # If True, convert None column values to empty strings in the samples.
        cache_limit: int | str | None='100mb', # The limit of the cache size in bytes. If None, no limit is applied.
        max_size: int | None=None, # Cuts the amount of samples to this size, if specified. Useful for debugging or testing.
        resolution: Any=None, # TODO: dirty hack to support the genvid repo...
        allow_missing_columns: bool=False, # If True, ignore missing columns in the index file.
        num_random_access_retries: int=5, # The number of retries to access a sample by its index.
        print_exceptions: bool=False, # If True, print exceptions in the main thread.
        print_traceback: bool=False, # If True, print the traceback of exceptions in the main thread.
        max_index_files_to_use: int | None=None, # If specified, use only the first N index files for the dataset. Useful for debugging or testing.
        lazy_index_chunk_size: int | None=None, # If positive, would only be reading `index_chunk_size` rows from the index file at a time. Also, won't load the whole index into memory.
        lazy_index_num_threads: int=3, # The number of threads to use for prefetching index chunks when using lazy index loading.
        lazy_index_prefetch_factor: int=3, # The number of index chunks to prefetch in the background when using lazy index loading.
        sql_query: str | None=None, # If specified, use the SQL query to filter/process the samples from the index file before downloading anything.
        min_num_pending_tasks_thresh: int | None=None, # The minimum number of pending tasks to keep in the downloader before scheduling more.
        unaligned_worker_index: bool = False, # Shall each worker iterate over the global dataset independently? Bad design, but helpful for tiny datasets.
        infinite_iteration: bool = False, # If True, the dataset would be iterated infinitely. Useful when you for some reason have batch_size > dataset_size and drop_last=True.

        # Some index optimization stuff.
        index_cols_to_keep: Sequence[str] | None=None, # Columns to keep in the index file. If None, all columns are kept.
    ):
        _ = resolution # Unused, kept for compatibility with the genvid repo.
        self.name: str = name if name is not None else os_utils.path_key(src, num_parts=3, drop_ext=True)
        self.src: str = src
        self.dst: str = dst
        self.data_type: DataSampleType | None = DataSampleType.from_str(data_type) if isinstance(data_type, str) else data_type
        self.shuffle_seed: int | None = build_shuffle_seed(shuffle_seed)
        self.transforms = transforms or []
        self.columns_to_download = columns_to_download
        self.num_downloading_workers = num_downloading_workers
        self.prefetch = prefetch
        self.num_downloading_retries = num_downloading_retries
        self.none_to_empty_str = none_to_empty_str
        self.index_col_name = index_col_name
        self._max_size = max_size
        self._node_cache_limit = os_utils.bytes_to_int(cache_limit) if cache_limit is not None else 0
        self._worker_cache_limit = None # Will be initialized later based on the number of workers per node.
        self._worker_disk_usage = 0 # Current cache usage in bytes.
        self._stored_sample_keys: deque[str] = deque() # A list of keys physicall stored on disk.
        self._allow_missing_columns = allow_missing_columns
        self._max_index_files_to_use = max_index_files_to_use
        self._lazy_index_chunk_size = lazy_index_chunk_size
        self._lazy_index_num_threads = lazy_index_num_threads
        self._lazy_index_prefetch_factor = lazy_index_prefetch_factor
        self._sql_query = sql_query
        self._min_num_pending_tasks_thresh: int = min_num_pending_tasks_thresh if min_num_pending_tasks_thresh is not None else MIN_NUM_PENDING_TASKS_THRESH[self.data_type]
        self._unaligned_worker_index = unaligned_worker_index
        self._infinite_iteration = infinite_iteration

        # Random access parameters.
        self._num_random_access_retries = num_random_access_retries
        self._print_exceptions = print_exceptions
        self._print_traceback = print_traceback
        if self._print_traceback:
            faulthandler.enable() # Printing all the traceback we can.

        # Some optimization parameters.
        self._gc = os_utils.TimeBasedGarbageCollector(interval_seconds=30)
        self._index_cols_to_keep = index_cols_to_keep

        assert self.index_col_name not in self.columns_to_download, f"Index column {self.index_col_name} cannot be in columns_to_download: {self.columns_to_download}."
        assert self.num_downloading_workers > 0, f"Number of workers must be greater than 0, but got {self.num_downloading_workers}."
        assert self.columns_to_download is not None and len(self.columns_to_download) > 0, f"Need to specify columns_to_download, but got {self.columns_to_download}."
        assert self._index_cols_to_keep is None or len(self._index_cols_to_keep) > 0, f"Must specify at least one column to keep in the index file, but got {self._index_cols_to_keep}."
        assert self._lazy_index_chunk_size is None or self._lazy_index_chunk_size >= 100, f"Lazy index chunk size must >= 100, but got {self._lazy_index_chunk_size}."
        assert not self._unaligned_worker_index or self._node_cache_limit > 0, f"Unaligned worker index requires caching to mitigate race conditions, but got {self._node_cache_limit}."

        self.epoch = 0
        self.sample_in_epoch = 0 # What sample idx we are in the current epoch.

        self.build_index() # Build the index metadata.
        self._index_partition = None # Index will be initialized in __iter__(), when we know the workers.
        self.downloader = self.init_downloader() # Initialize the downloader to download the shards in parallel.

    def init_downloader(self) -> ParallelDownloader:
        """Initializes a downloader to download the shards in parallel."""
        return ParallelDownloader(
            num_workers=self.num_downloading_workers,
            prefetch=self.prefetch,  # Prefetch a bit more than the number of workers.
            num_retries=self.num_downloading_retries,
            skip_if_exists=True,
        )

    def build_index(self):
        now = time.time()
        dist_utils.init_process_groups()
        if dist_utils.is_node_leader():
            logger.debug(f'[{self.name}] Building index on rank [{dist_utils.get_rank()}] for dataset {self.name} from source {self.src} to destination {self.dst}.')
            self.index_meta = build_index(
                src=self.src,
                dst_dir=self.dst,
                data_type=self.data_type,
                shuffle_seed=self.shuffle_seed,
                max_size=self._max_size,
                cols_to_keep=self._index_cols_to_keep,
                max_index_files_to_use=self._max_index_files_to_use,
                lazy=self._lazy_index_chunk_size is not None and self._lazy_index_chunk_size > 0,
                sql_query=self._sql_query,
            )
        else:
            logger.debug(f'[{self.name}] Waiting on rank [{dist_utils.get_rank()}] for the index to be built.')
            self.index_meta = None
        dist_utils.maybe_barrier()
        logger.debug(f'[{self.name}] Rank [{dist_utils.get_rank()}] finished building/waiting for building the index. Starting the broadcast.')
        self.index_meta = dist_utils.broadcast_object_locally(self.index_meta)
        logger.debug(f'[{self.name}] Got the index on rank [{dist_utils.get_rank()}]: {self.index_meta}. Took {time.time() - now:.2f} seconds.')

    def state_dict(self) -> dict[str, Any]:
        return {'epoch': self.epoch, 'sample_in_epoch': self.sample_in_epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.epoch = state_dict['epoch']
        self.sample_in_epoch = state_dict['sample_in_epoch']

    def __len__(self) -> int:
        return self.index_meta.num_samples

    def get_identifier_desc(self) -> str:
        """
        Returns a short description of the dataset object to store/lookup its FID/FVD/etc statistics.
        """
        transforms_desc_long = '-'.join([f'{type(t).__name__}-{str(sorted(vars(t)))}' for t in self.transforms])
        hasher = hashlib.sha256(transforms_desc_long.encode('utf-8'))
        transforms_hash = base64.urlsafe_b64encode(hasher.digest()).decode('ascii').rstrip('=')[:12]
        maxsize_str = f'maxsize{self._max_size}' if self._max_size is not None else ''
        return f'{self.name}{maxsize_str}-transforms-{transforms_hash}'

    @property
    def epoch_size(self) -> int:
        """
        Returns the size of the epoch, if specified, otherwise returns the total number of samples.
        """
        return self.index_meta.num_samples

    @beartype
    def set_progress(self, epoch: int, sample_in_epoch: int | None=None) -> None:
        self.epoch = epoch
        self.sample_in_epoch = sample_in_epoch if sample_in_epoch is not None else 0

    @staticmethod
    def partition_len(self) -> int:
        if self._index_partition is None:
            raise RuntimeError("Index is not initialized. Call __iter__() first.")
        return len(self._index_partition)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._safe_get_item(idx, num_retries_left=self._num_random_access_retries)

    def _safe_get_item(self, idx: int, num_retries_left: int=None) -> dict[str, Any]:
        try:
            return self._unsafe_get_item(idx)
        except Exception as e: # pylint: disable=broad-except
            if self._print_exceptions:
                logger.error(f"Exception in __getitem__({idx}): {e}")
            if self._print_traceback:
                traceback.print_exc()
            num_retries_left = self._num_random_access_retries if num_retries_left is None else num_retries_left
            if num_retries_left >= 0:
                new_idx = np.random.RandomState(idx).randint(low=0, high=len(self))
                return self._safe_get_item(idx=new_idx, num_retries_left=num_retries_left - 1)
            else:
                logger.error(f"Failed to load the video even after {self._num_random_access_retries} retries. Something is broken.")
                raise e

    def _unsafe_get_item(self, sample_id: int) -> dict[str, Any]:
        """
        Get sample by global index, blocking to download it.
        Note that for an intra-node index, we can only get samples from the current node-level partition only.
        """
        sample_meta = data_utils.read_parquet_slice(self.index_meta.path, start_offset=sample_id, end_offset=sample_id + 1).iloc[0].to_dict()
        total_size = self._schedule_download_(key=sample_id, sample_meta=sample_meta, blocking=True)
        sample_meta[SAMPLE_DISK_USAGE_FIELD] = total_size
        sample = next(self._construct_samples(sample_meta))
        # TODO: Ok, we never delete the samples obtained through random access queries:
        # - Reason 1. We hope that the user won't request too many randomly accessed samples since it's too slow anyway.
        # - Reason 2. This can break some race conditions with the downloader which assumes that it's sample space is not overlapping with anything else.
        # self._delete_sample_from_disk(sample_meta)
        return sample

    def _schedule_download_(self, key: int | str, sample_meta: dict[str, Any], blocking: bool=False) -> Any:
        columns_to_download = [col for col in self.columns_to_download if col in sample_meta and sample_meta[col] is not None and sample_meta[col] != '']
        assert self._allow_missing_columns or len(columns_to_download) == len(self.columns_to_download), \
            f"[{self.name}] Some columns are missing in the sample meta: {sample_meta}. Expected columns: {self.columns_to_download}, available columns: {columns_to_download}."
        assert len(columns_to_download) > 0, f"[{self.name}] No columns to download for sample {sample_meta}."
        source_urls: list[str] = [sample_meta[col] for col in columns_to_download]
        destinations: list[str] = [os.path.join(self.dst, self.name, f'{sample_meta[self.index_col_name]}-{col}{os_utils.file_ext(sample_meta[col]).lower()}') for col in columns_to_download]
        assert len(set(destinations)) == len(destinations), f"[{self.name}] Some destination paths are duplicated: {destinations}."
        downloading_result = self.downloader.schedule_task(
            key=key,
            source_urls=source_urls,
            destinations=destinations,
            blocking=blocking,
        )
        # Fill the sample_meta with the destination paths.
        for col, dst in zip(columns_to_download, destinations):
            sample_meta[col] = dst
        sample_meta[PROCESSED_FIELD] = True  # Mark the sample as processed.

        return downloading_result # Return smth meaningful only for blocking calls.

    def _schedule_downloads_(self, index_chunk: pd.DataFrame, scheduled_samples: dict[str, dict], shuffle_seed: int, epoch: int, global_worker_rank: int) -> int:
        schedule_order = misc.get_shuffled_sample_ids(len(index_chunk), shuffle_seed=shuffle_seed, epoch=epoch, rank=global_worker_rank)
        num_scheduled: int = 0
        for sample_id in schedule_order:
            sample_meta: dict[str, Any] = index_chunk.iloc[sample_id].to_dict()
            sample_key = sample_meta[self.index_col_name]
            if sample_key in scheduled_samples:
                logger.warning(f"[{self.name}] Sample {sample_key} is already scheduled, skipping. This likely means that we have duplicate IDs in the index file.")
                continue # Already scheduled.
            try:
                self._schedule_download_(key=sample_key, sample_meta=sample_meta, blocking=False)
                scheduled_samples[sample_key] = sample_meta  # Store the sample meta in the preallocated list.
                num_scheduled += 1
            except Exception as e:
                logger.error(f"[{self.name}] Failed to schedule download for sample {sample_meta}: {e}")
        logger.debug(f"Scheduled {num_scheduled} samples for download with {self.downloader}.")
        return num_scheduled

    def _construct_samples(self, sample_meta: dict[str, Any]) -> Iterator[SampleData]:
        # Loads all the binary files from the sample_meta.
        assert sample_meta[PROCESSED_FIELD], f"[{self.name}] Sample must be processed before loading."
        sample = {k: v for k, v in sample_meta.items() if k != PROCESSED_FIELD} # Creating a shallow copy of the sample_meta.

        # Augmenting with special keys.
        sample[SAMPLE_KEY_FIELD] = sample_meta[self.index_col_name]  # Add a key for the sample.
        if self.data_type is not None:
            sample[DATA_TYPE_FIELD] = str(self.data_type)  # Add the data type of the sample.

        # Some transforms may return multiple samples, so we yield from them instead of returning a single sample.
        yield from apply_transforms_recursively(sample, self.transforms)

    def _maybe_evict_cold_samples(self, scheduled_samples: dict[str, Any]):
        assert self._worker_cache_limit is not None, f"[{self.name}] Worker cache limit must be set before eviction."
        num_failures = 0
        while self._worker_disk_usage > self._worker_cache_limit:
            try:
                assert len(self._stored_sample_keys) > 0, f"[{self.name}] The state has diverged, no samples to evict. Disk usage: {self._worker_disk_usage}, cache limit: {self._worker_cache_limit}."
                sample_key = self._stored_sample_keys.popleft() # Get the oldest sample key to evict.
                assert sample_key in scheduled_samples, f"[{self.name}] Sample key {sample_key} not found in scheduled_samples."
                self._delete_sample_from_disk(scheduled_samples[sample_key])
                self._worker_disk_usage -= scheduled_samples[sample_key][SAMPLE_DISK_USAGE_FIELD]
                del scheduled_samples[sample_key]  # Remove the sample from the processed samples.
            except Exception as e:
                logger.error(f"[{self.name}] Failed to evict sample: {e}")
                num_failures += 1
                if num_failures > 100:
                    break # There is something wrong with the eviction process, stop it to avoid infinite loop.

    def _maybe_init_worker_cache_limit(self, global_worker_rank: int, global_num_workers: int) -> None:
        if self._worker_cache_limit is not None:
            return
        num_gpu_ranks = dist_utils.get_world_size()
        assert global_num_workers % num_gpu_ranks == 0, f"Each GPU is expected to have the same amount of DL workers. Found {global_num_workers} workers for {num_gpu_ranks} GPUs."
        num_workers_per_node = global_num_workers // dist_utils.get_num_nodes()
        self._worker_cache_limit = self._node_cache_limit // num_workers_per_node
        logger.debug(f"[{self.name}] Initialized worker cache limit: {self._worker_cache_limit} bytes for rank {global_worker_rank} with {global_num_workers} workers (num_workers_per_node={num_workers_per_node}, num_nodes={dist_utils.get_num_nodes()}).")

    def _delete_sample_from_disk(self, sample_meta: dict[str, Any]) -> None:
        assert sample_meta[PROCESSED_FIELD], f"[{self.name}] Sample must be processed before deletion."
        for col in self.columns_to_download:
            try:
                assert col in sample_meta, f"[{self.name}] Column {col} not found in sample_meta with keys: {list(sample_meta.keys())}."
                file_path = sample_meta[col]
                if file_path in ('', None):
                    assert self._allow_missing_columns, f"[{self.name}] Column {col} is empty in sample_meta: {sample_meta}, while allow_missing_columns is False."
                    continue
                assert os.path.exists(file_path), f"[{self.name}] File {file_path} does not exist."
                os.remove(file_path)
            except Exception as e:
                logger.error(f"[{self.name}] Column: {col}. Failed to delete file {file_path} for sample {sample_meta[self.index_col_name]}: {e}")
                if self._print_exceptions:
                    logger.error(traceback.format_exc())

    def __del__(self):
        if hasattr(self, 'downloader') and self.downloader is not None:
            self.downloader.shutdown()

    def _reset(self):
        if self.downloader.thread_pool is None:
            self.downloader.init_thread_pool()
        else:
            self.downloader.reset()
            self.downloader.stop() # Shutting down the previous downloader in an async way for its workers to finish and die.
            self.downloader = self.init_downloader()  # Reinitialize the downloader to reset its state.
            self._worker_disk_usage = 0
            self._stored_sample_keys.clear()

    def _iter_chunks_(self, index_iterator: Iterator[pd.DataFrame], global_worker_rank: int, scheduled_samples: dict[str, dict[str, Any]]) -> Iterator[dict[str, Any]]:
        logger.debug(f'[{self.name} | rank {dist_utils.get_rank()} | worker_rank {global_worker_rank}/{dist_utils.get_world_size()}] Starting to iterate over the dataset {self.name}. Epoch: {self.epoch}, sample_in_epoch: {self.sample_in_epoch}.')
        scheduling_kwargs = dict(scheduled_samples=scheduled_samples, shuffle_seed=self.shuffle_seed, epoch=self.epoch, global_worker_rank=global_worker_rank)
        self._schedule_downloads_(next(index_iterator), **scheduling_kwargs)

        for sample_key, (total_sample_size, total_download_size) in self.downloader.yield_completed():
            self._stored_sample_keys.append(sample_key)
            self._worker_disk_usage += total_download_size
            scheduled_samples[sample_key][SAMPLE_DISK_USAGE_FIELD] = total_sample_size

            try:
                yield from self._construct_samples(scheduled_samples[sample_key])
            except Exception as e:
                logger.error(f"[{self.name}] Failed to construct samples from {sample_key}: {e}")
                if self._print_traceback:
                    logger.error(traceback.format_exc())
            self._maybe_evict_cold_samples(scheduled_samples)
            self._gc.maybe_collect()

            # We need to increment the sample index regardless of whether the sample was processed successfully or not.
            # Otherwise, we would start at a much earlier actual sample id when loading the state dict.
            self.sample_in_epoch += 1

            # If we have less than the min amount of pending tasks, schedule more.
            if self.downloader.get_num_pending_tasks() < self._min_num_pending_tasks_thresh and (next_index_chunk := next(index_iterator, None)) is not None:
                self._schedule_downloads_(next_index_chunk, **scheduling_kwargs)

    def __iter_epoch__(self) -> Iterator[dict[str, Any]]:
        self._reset() # TODO: not sure if this resetting is correct...

        global_worker_rank, global_num_workers = dist_utils.get_global_worker_info()
        assert len(self) >= global_num_workers or self._unaligned_worker_index, f"Dataset size {len(self)} is smaller than the number of dataloading workers {global_num_workers} while unaligned_worker_index=False."
        self._maybe_init_worker_cache_limit(global_worker_rank, global_num_workers)
        scheduled_samples: dict[str, dict[str, Any]] = {}

        if self.index_meta.lazy:
            index_chunks_it = LazyIndexIterator(
                path=self.index_meta.path,
                total_num_samples=min(self.index_meta.num_samples, self._max_size or float('inf')),
                chunk_size=self._lazy_index_chunk_size,
                shuffle_seed=self.shuffle_seed,
                epoch=self.epoch,
                sample_in_epoch=self.sample_in_epoch,
                num_threads=self._lazy_index_num_threads,
                prefetch_factor=self._lazy_index_prefetch_factor,
                sql_query=self._sql_query,
                unaligned_worker_index=self._unaligned_worker_index,
            )
        else:
            if self._index_partition is None:
                partition_rank, partition_num_workers = (0, 1) if self._unaligned_worker_index else (global_worker_rank, global_num_workers)
                self._index_partition = load_index_partition(self.index_meta, partition_rank, partition_num_workers, dist_utils.get_num_nodes())
            index_chunks_it = lean_index_iterator(
                self._index_partition, chunk_size=SCHEDULE_BATCH_SIZE, sample_in_epoch=self.sample_in_epoch,
                shuffle_seed=self.shuffle_seed, epoch=self.epoch, rank=global_worker_rank)

        yield from self._iter_chunks_(
            index_iterator=index_chunks_it,
            global_worker_rank=global_worker_rank,
            scheduled_samples=scheduled_samples,
        )

        logger.debug(f"Processed {self.sample_in_epoch} samples in epoch {self.epoch}.")
        self.sample_in_epoch = 0  # Reset the sample index for the next epoch.
        self.epoch += 1 # TODO: this would be incrementing the epoch each time a new dataloader is called over the dataset, which is not good.

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while True:
            yield from self.__iter_epoch__()
            if self._infinite_iteration:
                logger.debug(f"[{self.name}] Starting a new epoch {self.epoch} since self._infinite_iteration=True.")
            else:
                break

#----------------------------------------------------------------------------
# Index iteration utils.

class LazyIndexIterator:
    """
    An iterator that fetches chunks of a lazy index in parallel using a thread pool.

    This class calculates the chunks of the index that belong to a specific worker
    in a distributed environment, shuffles them, and then uses a background thread
    pool to fetch the actual data for these chunks ahead of time.
    """
    def __init__(
        self,
        path: str, # Path to the index file.
        total_num_samples: int,
        chunk_size: int,
        shuffle_seed: int,
        epoch: int,
        sample_in_epoch: int = 0, # The sample index in the current epoch, used for slicing.
        num_threads: int = 4,
        prefetch_factor: int = 5,
        sql_query: str | None = None, # SQL query to filter/process the index before fetching chunks.
        unaligned_worker_index: bool = False,
    ):
        self.path = path
        self.sql_query = sql_query
        self._pool = LazyThreadPool(num_workers=num_threads, prefetch=num_threads * prefetch_factor)

        # --- Calculate chunking for the current worker ---
        self.global_worker_rank, self.global_num_workers = dist_utils.get_global_worker_info()
        partition_rank, partition_num_workers = (0, 1) if unaligned_worker_index else (self.global_worker_rank, self.global_num_workers)
        num_samples_per_worker = total_num_samples // partition_num_workers
        num_chunks = (num_samples_per_worker + chunk_size - 1) // chunk_size
        self.chunk_size_refined = math.floor(num_samples_per_worker / num_chunks)
        self.partition_start_sample_id = partition_rank * num_samples_per_worker

        # --- Determine the order of chunks to process ---
        chunk_order = misc.get_shuffled_sample_ids(num_chunks, shuffle_seed=shuffle_seed, epoch=epoch, rank=self.global_worker_rank)

        # --- Schedule all fetching tasks immediately ---
        sample_offset = 0
        for chunk_id in chunk_order:
            start_sample_id = chunk_id * self.chunk_size_refined + self.partition_start_sample_id
            end_sample_id = min(start_sample_id + self.chunk_size_refined, total_num_samples)
            cur_chunk_size = end_sample_id - start_sample_id
            if sample_in_epoch < sample_offset + cur_chunk_size:
                # Only schedule chunks that are after the current sample in the epoch.
                start_sample_id += max(0, sample_in_epoch - sample_offset)
                self._pool.schedule_task(self._fetch_chunk, task_input={'start': start_sample_id, 'end': end_sample_id})
            else:
                logger.debug(f'[rank {dist_utils.get_rank()} | worker_rank {self.global_worker_rank}/{self.global_num_workers}] Skipping chunk [{start_sample_id}:{end_sample_id}] as it is before sample_in_epoch {sample_in_epoch}.')
            sample_offset += cur_chunk_size

    def _fetch_chunk(self, task_input: dict[str, int]) -> pd.DataFrame:
        start_id = task_input['start']
        end_id = task_input['end']
        logger.debug(f"[rank {dist_utils.get_rank()} | worker_rank {self.global_worker_rank}/{self.global_num_workers}] Fetching index chunk from {self.path} [{start_id}:{end_id}]")
        index_slice = data_utils.read_parquet_slice(self.path, start_offset=start_id, end_offset=end_id)
        index_slice = data_utils.maybe_run_sql_query_on_dataframe(index_slice, sql_query=self.sql_query)
        logger.debug(f"[rank {dist_utils.get_rank()} | worker_rank {self.global_worker_rank}/{self.global_num_workers}] Fetched index chunk from {self.path} [{start_id}:{end_id}], shape: {index_slice.shape}")
        return index_slice

    def __iter__(self) -> Iterator[pd.DataFrame]:
        return self

    def __next__(self) -> pd.DataFrame:
        """Returns the next prefetched index chunk. Blocks until a chunk is available."""
        try:
            for task_result in self._pool.yield_completed():
                if task_result['success']:
                    if task_result['task_output'].empty:
                        logger.warning(f"Index slice is empty for {self.path} [{task_result['task_input']['start']}:{task_result['task_input']['end']}]. This might be due to an empty index or a SQL query that filtered out all rows.")
                        continue
                    return task_result['task_output']
                else:
                    # Propagate the error from the worker thread to the main thread
                    logger.error(f"Failed to prefetch index chunk: {task_result['error']}")
                    continue
        except StopIteration:
            # All tasks are done, ensure the pool is shut down before we stop
            self.shutdown()
            raise

    def shutdown(self):
        if self._pool:
            logger.debug("Shutting down LazyIndexIterator's thread pool.")
            self._pool.shutdown()
            self._pool = None

    def __del__(self):
        if hasattr(self, '_pool') and self._pool is not None:
            self.shutdown()

def lean_index_iterator(index: pd.DataFrame, chunk_size: int, sample_in_epoch: int, **shuffling_kwargs) -> Iterator[pd.DataFrame]:
    """An iterator that yields index chunks one by one. Convenint for lean scheduling."""
    sample_order = misc.get_shuffled_sample_ids(len(index), **shuffling_kwargs)
    sample_order = sample_order[sample_in_epoch:]  # Skip samples that are already processed in the current epoch.
    num_chunks = (len(sample_order) + chunk_size - 1) // chunk_size

    for chunk_id in range(num_chunks):
        cur_sample_ids = sample_order[chunk_id * chunk_size : (chunk_id + 1) * chunk_size]
        yield index.iloc[cur_sample_ids]

#----------------------------------------------------------------------------
# Transforms utils.

def apply_transforms_recursively(sample: SampleData, transforms: list[SampleTransform]) -> Iterator[SampleData]:
    """
    Applies a list of transforms to a sample sequentially. A transform can return
    a single sample or an iterator/generator of samples.
    """
    assert isinstance(sample, dict), f"Sample must be a dictionary, but got {type(sample)}."

    # Base case: If there are no more transforms, yield the processed sample.
    if not transforms:
        yield sample
        return

    # Apply the current transform to the sample.
    result = transforms[0](sample)

    # The result might be a single new sample or a generator/iterator of new samples.
    if isinstance(result, types.GeneratorType):
        for processed_sample in result:
            # For each sample yielded by the transform, apply the rest of the transforms.
            yield from apply_transforms_recursively(processed_sample, transforms[1:])
    # It might also be another iterable like a list, but not a dict (which is our sample type)
    elif isinstance(result, (list, tuple)):
        for processed_sample in result:
             yield from apply_transforms_recursively(processed_sample, transforms[1:])
    # Otherwise, it's a single sample.
    else:
        yield from apply_transforms_recursively(result, transforms[1:])

#----------------------------------------------------------------------------

@beartype
def build_shuffle_seed(seed: int | None) -> int | None:
    """
    If
    """
    if seed is None or seed >= 0: return seed

    assert seed == -1, f"Invalid shuffle seed: {seed}. Must be None, -1, or a non-negative integer."
    seed = int(hashlib.sha256(os.urandom(16)).hexdigest(), 16) % (2 ** 32)
    if dist_utils.is_main_process():
        logger.info(f"Broadcasting a random shuffle seed: {seed} across all ranks.")
    seed = dist_utils.broadcast_object(seed, src=0)
    return seed

#----------------------------------------------------------------------------
