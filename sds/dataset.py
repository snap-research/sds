import os
import time
import types
from typing import Any, Iterator
from collections import deque
from collections.abc import Callable
from threading import Event
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from torch.utils.data import IterableDataset
from loguru import logger

from sds.downloader import ParallelDownloader
from sds.structs import DataSampleType, SampleData, SampleTransform
from sds.index import build_index, load_index_slice, load_index_row
import sds.utils.distributed as dist_utils
import sds.utils.os_utils as os_utils

#---------------------------------------------------------------------------
# Special fields names.

SAMPLE_DISK_USAGE_FIELD = '__disk_usage__' # The total size of the sample in bytes.
PROCESSED_FIELD = '__is_processed__' # A flag to mark the sample as processed.
SAMPLE_KEY_FIELD = '__sample_key__' # A key for the sample, corresponding to the index column value.

#---------------------------------------------------------------------------

class StreamingDataset(IterableDataset):
    def __init__(self,
        src: str, # a CSV file path, a JSON file path, or a directory path (possibly remote)
        dst: str, # A local directory path where to store the samples
        data_type: DataSampleType | str, # The type of the dataset, e.g. 'csv', 'json', 'parquet', or 'directory'
        # local_shm_path: str, # A local file system path which only the workers of the current node can access.
        # global_shm_path: str, # A global file system path which any rank can access globally.
        shuffle_seed: int | None=None, # Shuffle seed for the dataset.
        transforms: list[Callable]=None, # A list of data augmentation callbacks to apply to the samples.
        columns_to_download: list[str] | None=None, # The names of the columns to use from the index file.
        index_col_name: str='index',
        num_downloading_workers: int=4, # The number of workers to use for downloading the samples in parallel.
        prefetch: int=10, # The number of samples to prefetch in the downloader.
        num_downloading_retries: int=3, # The number of retries to download a sample if it fails.
        none_to_empty_str: bool=True, # If True, convert None column values to empty strings in the samples.
        cache_limit: str | None='100mb', # The limit of the cache size in bytes. If None, no limit is applied.
    ):
        self.name: str = os_utils.file_key(src)
        self.src: str = src
        self.dst: str = dst
        self.data_type: DataSampleType = DataSampleType.from_str(data_type) if isinstance(data_type, str) else data_type
        self.shuffle_seed: int | None = shuffle_seed
        self.transforms = transforms or []
        self.columns_to_download = columns_to_download
        self.num_downloading_workers = num_downloading_workers
        self.prefetch = prefetch
        self.num_downloading_retries = num_downloading_retries
        self.none_to_empty_str = none_to_empty_str
        self.index_col_name = index_col_name
        self._node_cache_limit = os_utils.bytes_to_int(cache_limit)
        self._worker_cache_limit = None
        self._disk_usage = 0 # Current cache usage in bytes.
        self._stored_sample_ids: deque[int] = deque() # A list of keys physicall stored on disk.
        self._gc = os_utils.TimeBasedGarbageCollector(interval_seconds=30)

        assert self.index_col_name not in self.columns_to_download, f"Index column {self.index_col_name} cannot be in columns_to_download: {self.columns_to_download}."
        assert self.num_downloading_workers > 0, f"Number of workers must be greater than 0, but got {self.num_downloading_workers}."
        assert self.columns_to_download is not None and len(self.columns_to_download) > 0, f"Need to specify columns_to_download, but got {self.columns_to_download}."
        assert self._node_cache_limit > 100_000_000, f"Cache limit {self._node_cache_limit} is too small, must be at least 100MB."

        self.epoch = -1
        self.num_yielded_samples = 0

        self.build_index() # Build the index metadata. TODO: it's slow sometimes, so maybe we should to it lazily.
        self.index_slice = None # Index will be initialized in __iter__(), when we know the workers.

        self._executor: ThreadPoolExecutor | None = None # A background thread to do call downloading/deletion to not block the main thread.
        self._crash_event: Event | None = None  # An event to signal if the background thread has crashed.

        # A downloader to download the shards in parallel.
        self.downloader = ParallelDownloader(
            num_workers=self.num_downloading_workers,
            prefetch=self.prefetch,  # Prefetch a bit more than the number of workers.
            num_retries=self.num_downloading_retries,
            skip_if_exists=True,
        )

    def build_index(self):
        now = time.time()
        logger.debug('Building index...')
        self.index_meta = build_index(self.src, self.dst, self.data_type, self.shuffle_seed) if dist_utils.is_node_leader() else None
        dist_utils.maybe_barrier()
        self.index_meta = dist_utils.broadcast_object_locally(self.index_meta)
        logger.debug(f'Constructed an index: {self.index_meta}. Took {time.time() - now:.2f} seconds.')

    def state_dict(self) -> dict[str, Any]:
        return {'epoch': self.epoch, 'num_yielded_samples': self.num_yielded_samples}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.epoch = state_dict['epoch']
        self.num_yielded_samples = state_dict['num_yielded_samples']

    def __len__(self) -> int:
        return self.index_meta.num_samples

    @staticmethod
    def partition_len(self) -> int:
        if self.index_slice is None:
            raise RuntimeError("Index is not initialized. Call __iter__() first.")
        return len(self.index_slice)

    def __getitem__(self, sample_id: int) -> dict[str, Any]:
        """
        Get sample by global index, blocking to download it.
        Note that for an intra-node index, we can only get samples from the current node-level partition only.
        """
        sample_meta = load_index_row(self.index_meta, sample_id).iloc[0].to_dict()
        total_size = self._schedule_download_(key=sample_id, sample_meta=sample_meta, blocking=True)
        sample_meta[SAMPLE_DISK_USAGE_FIELD] = total_size
        # TODO: we need to store self._processed_sample_metas as an in-class property to be able to consider it for eviction.
        # self._stored_sample_ids.append(sample_id)
        # self._disk_usage += total_size
        return next(self._construct_samples(sample_meta))

    def _schedule_download_(self, key: int | str, sample_meta: dict[str, Any], blocking: bool=False) -> Any:
        source_urls: list[str] = [sample_meta[col] for col in self.columns_to_download]
        destinations: list[str] = [os.path.join(self.dst, self.name, sample_meta[self.index_col_name] + os_utils.file_ext(sample_meta[col]).lower()) for col in self.columns_to_download]
        downloading_result = self.downloader.schedule_task(
            key=key,
            source_urls=source_urls,
            destinations=destinations,
            blocking=blocking,
        )
        # Fill the sample_meta with the destination paths.
        for col, dst in zip(self.columns_to_download, destinations):
            sample_meta[col] = dst
        sample_meta[PROCESSED_FIELD] = True  # Mark the sample as processed.

        return downloading_result # Return smth meaningful only for blocking calls.

    def _schedule_downloads(self, sample_ids: list[int]) -> list[dict[str, Any]]:
        processed_sample_metas: list[dict[str, Any]] = [None] * len(sample_ids)  # Preallocate a list to store processed sample metas.
        for sample_id in sample_ids:
            sample_meta: dict[str, Any] = self.index_slice.iloc[sample_id].to_dict()
            self._schedule_download_(key=sample_id, sample_meta=sample_meta, blocking=False)
            processed_sample_metas[sample_id] = sample_meta  # Store the sample meta in the preallocated list.
        logger.debug(f"Scheduled {len(processed_sample_metas)} samples for download with {self.downloader}.")
        return processed_sample_metas

    def _construct_samples(self, sample_meta: dict[str, Any]) -> Iterator[SampleData]:
        # Loads all the binary files from the sample_meta.
        assert sample_meta[PROCESSED_FIELD], "Sample must be processed before loading."
        sample = {k: v for k, v in sample_meta.items() if k != PROCESSED_FIELD} # Creating a shallow copy of the sample_meta.

        # Augmenting with special keys.
        sample[SAMPLE_KEY_FIELD] = sample_meta[self.index_col_name]  # Add a key for the sample.

        # Some transforms may return multiple samples, so we yield from them instead of returning a single sample.
        yield from apply_transforms_recursively(sample, self.transforms)

    def _maybe_evict_cold_samples(self, processed_sample_metas: dict[str, Any]) -> bool:
        assert self._worker_cache_limit is not None, "Worker cache limit must be set before eviction."
        while self._disk_usage > self._worker_cache_limit:
            assert len(self._stored_sample_ids) > 0, f"The state has diverged, no samples to evict. Disk usage: {self._disk_usage}, cache limit: {self._worker_cache_limit}."
            sample_id = self._stored_sample_ids.popleft() # Get the oldest sample key to evict.
            self._delete_sample_from_disk(processed_sample_metas[sample_id])
            self._disk_usage -= processed_sample_metas[sample_id][SAMPLE_DISK_USAGE_FIELD]
            logger.debug(f"Current disk usage: {self._disk_usage} bytes, cache limit: {self._worker_cache_limit} bytes.")

    def _maybe_init_worker_cache_limit(self, dl_worker_rank: int, dl_num_workers: int) -> None:
        if self._worker_cache_limit is not None:
            return
        num_gpu_ranks = dist_utils.get_world_size()
        assert dl_num_workers % num_gpu_ranks == 0, f"Each GPU is expected to have the same amount of DL workers. Found {dl_num_workers} workers for {num_gpu_ranks} GPUs."
        num_workers_per_node = dl_num_workers // dist_utils.get_num_nodes()
        self._worker_cache_limit = self._node_cache_limit // num_workers_per_node
        logger.debug(f"Initialized worker cache limit: {self._worker_cache_limit} bytes for rank {dl_worker_rank} with {dl_num_workers} workers (num_workers_per_node={num_workers_per_node}, num_nodes={dist_utils.get_num_nodes()}).")

    def _delete_sample_from_disk(self, sample_meta: dict[str, Any]) -> None:
        assert sample_meta[PROCESSED_FIELD], "Sample must be processed before deletion."
        for col in self.columns_to_download:
            file_path = sample_meta[col]
            assert os.path.exists(file_path), f"File {file_path} does not exist."
            os.remove(file_path)
            logger.debug(f"Deleted file {file_path} for sample {sample_meta[self.index_col_name]}.")

    def __del__(self):
        if self.downloader is not None:
            self.downloader.shutdown()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.downloader.thread_pool is None:
            self.downloader.init_thread_pool()
        else:
            self.downloader.clear_pending_downloads()
            # TODO: this looks like a bug.
            self._disk_usage = 0
            self._stored_sample_ids.clear()

        dl_worker_rank, dl_num_workers = dist_utils.get_safe_worker_info()
        self._maybe_init_worker_cache_limit(dl_worker_rank, dl_num_workers)
        if self.index_slice is None:
            self.index_slice = load_index_slice(self.index_meta, dl_worker_rank, dl_num_workers, dist_utils.get_num_nodes())
        self.epoch += 1 # TODO: this would be incrementing the epoch each time a new dataloader is called over the dataset, which is not good.

        if self._crash_event is None:
            self._crash_event = Event()
        elif self._crash_event.is_set():
            raise RuntimeError('Background thread failed. Check other traceback.')

        # Creating a list of sample IDs to iterate over. Assuming that they will fit in memory.
        sample_ids = list(range(len(self.index_slice)))
        if self.shuffle_seed is not None:
            cur_seed = (self.shuffle_seed * 1_000_003 + self.epoch * 1_000_037 + dl_worker_rank * 1_000_039) % (2**32)
            sample_ids = np.random.RandomState(cur_seed).permutation(sample_ids).tolist()

        # worker_sample_id_offset = dl_worker_rank * (len(sample_ids) // dl_num_workers)
        # sample_ids = sample_ids[worker_sample_id_offset:]
        processed_sample_metas: list[dict[str, Any]] = self._schedule_downloads(sample_ids)

        for sample_id, total_size in self.downloader.yield_completed_keys():
            processed_sample_metas[sample_id][SAMPLE_DISK_USAGE_FIELD] = total_size
            self._stored_sample_ids.append(sample_id)
            self._disk_usage += total_size
            yield from self._construct_samples(processed_sample_metas[sample_id])
            self._maybe_evict_cold_samples(processed_sample_metas)
            self._gc.maybe_collect()
            self.num_yielded_samples += 1
        logger.debug(f"Yielded {self.num_yielded_samples} samples in epoch {self.epoch}.")
        self.downloader.wait_completion()

#----------------------------------------------------------------------------

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
