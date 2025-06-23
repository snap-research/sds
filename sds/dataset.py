import os
from typing import Any, Iterator
from collections import deque
from collections.abc import Callable
from threading import Event
from concurrent.futures import ThreadPoolExecutor

from torch.utils.data import IterableDataset
from loguru import logger
from streaming.base.util import bytes_to_int

from sds.downloader import ParallelDownloader
from sds.utils.misc import pseudo_shuffle
from sds.structs import DataSampleType
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
        data_type: DataSampleType, # The type of the dataset, e.g. 'csv', 'json', 'parquet', or 'directory'
        # local_shm_path: str, # A local file system path which only the workers of the current node can access.
        # global_shm_path: str, # A global file system path which any rank can access globally.
        shuffle_seed: int | None=None, # Shuffle seed for the dataset.
        data_processing_callbacks: list[Callable]=None, # A list of data augmentation callbacks to apply to the samples.
        columns_to_yield: list[str] | None=None, # The names of the columns to use from the index file.
        columns_to_load: list[str] | None=None, # The names of the columns to use from the index file.
        index_col_name: str='index',
        num_downloading_workers: int=4, # The number of workers to use for downloading the samples in parallel.
        none_to_empty_str: bool=True, # If True, convert None column values to empty strings in the samples.
        cache_limit: str | None='100mb', # The limit of the cache size in bytes. If None, no limit is applied.
    ):
        self.name: str = os_utils.file_key(src)
        self.src: str = src
        self.dst: str = dst
        self.data_type: DataSampleType = data_type
        self.shuffle_seed: int | None = shuffle_seed
        self.data_processing_callbacks = data_processing_callbacks or []
        self.columns_to_yield = columns_to_yield
        self.columns_to_load = columns_to_load
        self.num_downloading_workers = num_downloading_workers
        self.none_to_empty_str = none_to_empty_str
        self.index_col_name = index_col_name
        self._cache_limit = bytes_to_int(cache_limit)
        self._disk_usage = 0 # Current cache usage in bytes.
        self._stored_sample_ids: deque[int] = deque() # A list of keys physicall stored on disk.

        assert self.index_col_name not in self.columns_to_load, f"Index column {self.index_col_name} cannot be in columns_to_load: {self.columns_to_load}."
        assert self.num_downloading_workers > 0, f"Number of workers must be greater than 0, but got {self.num_downloading_workers}."
        assert self.columns_to_load is not None and len(self.columns_to_load) > 0, f"Need to specify columns_to_load, but got {self.columns_to_load}."
        assert self._cache_limit > 100_000_000, f"Cache limit {self._cache_limit} is too small, must be at least 100MB."

        self.index_meta = build_index(src, dst, self.data_type, self.shuffle_seed) if dist_utils.is_node_leader() else None
        dist_utils.maybe_barrier()
        self.index_meta = dist_utils.broadcast_object_locally(self.index_meta)
        self.epoch = -1
        self.num_samples_yielded = 0
        logger.debug(f'Constructed an index: {self.index_meta}')

        # Index will be initialized in __iter__(), when we know the workers.
        self.index = None

        self._executor: ThreadPoolExecutor | None = None # A background thread to do call downloading/deletion to not block the main thread.
        self._crash_event: Event | None = None  # An event to signal if the background thread has crashed.

        # A downloader to download the shards in parallel.
        self.downloader = ParallelDownloader(
            num_workers=self.num_downloading_workers,
            prefetch=self.num_downloading_workers * 10,  # Prefetch a bit more than the number of workers.
            num_retries=3,
            skip_if_exists=True,
        )

    def state_dict(self) -> dict[str, Any]:
        return {'epoch': self.epoch, 'num_samples_yielded': self.num_samples_yielded}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.epoch = state_dict['epoch']
        self.num_samples_yielded = state_dict['num_samples_yielded']

    def __len__(self) -> int:
        return self.index_meta.num_samples

    @staticmethod
    def partition_len(self) -> int:
        if self.index is None:
            raise RuntimeError("Index is not initialized. Call __iter__() first.")
        return len(self.index)

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
        return self._load_sample(sample_meta)

    def _schedule_download_(self, key: int | str, sample_meta: dict[str, Any], blocking: bool=False) -> Any:
        source_urls: list[str] = [sample_meta[col] for col in self.columns_to_load]
        destinations: list[str] = [os.path.join(self.dst, self.name, sample_meta[self.index_col_name] + os_utils.file_ext(sample_meta[col])) for col in self.columns_to_load]
        downloading_result = self.downloader.schedule_task(
            key=key,
            source_urls=source_urls,
            destinations=destinations,
            blocking=blocking,
        )
        # Fill the sample_meta with the destination paths.
        for col, dst in zip(self.columns_to_load, destinations):
            sample_meta[col] = dst
        sample_meta[PROCESSED_FIELD] = True  # Mark the sample as processed.

        return downloading_result # Return smth meaningful only for blocking calls.

    def _schedule_downloads(self, sample_ids: list[int]) -> list[dict[str, Any]]:
        processed_sample_metas: list[dict[str, Any]] = []
        for sample_id in sample_ids:
            sample_meta: dict[str, Any] = self.index.iloc[sample_id].to_dict()
            self._schedule_download_(key=sample_id, sample_meta=sample_meta, blocking=False)
            processed_sample_metas.append(sample_meta)
        logger.debug(f"Scheduled {len(processed_sample_metas)} samples for download with {self.downloader}.")
        return processed_sample_metas

    def _load_sample(self, sample_meta: dict[str, Any]) -> dict[str, Any]:
        # Loads all the binary files from the sample_meta.
        assert sample_meta[PROCESSED_FIELD], "Sample must be processed before loading."
        sample_data = {}

        # Loading the downloaded binary files into memory. TODO: we can move this into callbacks.
        columns_to_yield = self.columns_to_yield if self.columns_to_yield is not None else list(sample_meta.keys())
        for col in columns_to_yield:
            if col in self.columns_to_load:
                assert col in sample_meta, f"Column {col} not found in sample_meta."
                with open(sample_meta[col], 'rb') as f:
                    sample_data[col] = f.read()
            else:
                sample_data[col] = sample_meta[col]

        # Augmenting with special keys.
        sample_data[self.index_col_name] = sample_meta[self.index_col_name]  # Add the index column.
        sample_data[SAMPLE_KEY_FIELD] = sample_meta[self.index_col_name]  # Add a key for the sample.

        # Apply the data processing callbacks.
        for callback in self.data_processing_callbacks:
            sample_data = callback(sample_data)

        return sample_data

    def _maybe_evict_cold_samples(self, processed_sample_metas: dict[str, Any]) -> bool:
        while self._disk_usage > self._cache_limit:
            assert len(self._stored_sample_ids) > 0, f"The state has diverged, no samples to evict. Disk usage: {self._disk_usage}, cache limit: {self._cache_limit}."
            sample_id = self._stored_sample_ids.popleft() # Get the oldest sample key to evict.
            self._delete_sample_from_disk(processed_sample_metas[sample_id])
            self._disk_usage -= processed_sample_metas[sample_id][SAMPLE_DISK_USAGE_FIELD]
            logger.debug(f"Current disk usage: {self._disk_usage} bytes, cache limit: {self._cache_limit} bytes.")

    def _delete_sample_from_disk(self, sample_meta: dict[str, Any]) -> None:
        assert sample_meta[PROCESSED_FIELD], "Sample must be processed before deletion."
        for col in self.columns_to_load:
            file_path = sample_meta[col]
            assert os.path.exists(file_path), f"File {file_path} does not exist."
            os.remove(file_path)
            logger.debug(f"Deleted file {file_path} for sample {sample_meta[self.index_col_name]}.")

    def __del__(self):
        if self.downloader is not None:
            self.downloader.shutdown()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.downloader is not None:
            self.downloader.clear_pending_downloads()
            # TODO: this looks like a bug.
            self._disk_usage = 0
            self._stored_sample_ids.clear()

        dl_worker_rank, dl_num_workers = dist_utils.get_safe_worker_info()
        if self.index is None:
            self.index = load_index_slice(self.index_meta, dl_worker_rank, dl_num_workers)
        self.epoch += 1 # TODO: this would be incrementing the epoch each time a new dataloader is called over the dataset, which is not good.

        if self._crash_event is None:
            self._crash_event = Event()
        elif self._crash_event.is_set():
            raise RuntimeError('Background thread failed. Check other traceback.')

        # Creating a list of sample IDs to iterate over. Assuming that they will fit in memory.
        if self.shuffle_seed is not None:
            cur_seed = (self.shuffle_seed * 1_000_003 + self.epoch * 1_000_037 + self.worker * 1_000_039) % (2**32)
            sample_ids = pseudo_shuffle(len(self.index), cur_seed)
        else:
            sample_ids = list(range(len(self.index)))

        processed_sample_metas: list[dict[str, Any]] = self._schedule_downloads(sample_ids)

        for sample_id, total_size in self.downloader.yield_completed_keys():
            logger.debug(f"Processing sample {sample_id}...")
            processed_sample_metas[sample_id][SAMPLE_DISK_USAGE_FIELD] = total_size
            self._stored_sample_ids.append(sample_id)
            self._disk_usage += total_size
            yield self._load_sample(processed_sample_metas[sample_id])
            self._maybe_evict_cold_samples(processed_sample_metas)
        logger.debug(f"Yielded {self.num_samples_yielded} samples in epoch {self.epoch}.")
        self.downloader.wait_completion()

#----------------------------------------------------------------------------
