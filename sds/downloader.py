import os
import urllib.parse
from typing import Generator
from dataclasses import dataclass

from loguru import logger
from sds.utils.download import CloudDownloader

from sds.lazy_thread_pool import LazyThreadPool

#----------------------------------------------------------------------------

@dataclass
class DownloadingTask:
    key: str
    source_urls: list[str]
    destinations: list[str]
    timeout: int # Timeout in seconds.
    downloaders: dict[str, CloudDownloader]
    skip_if_exists: bool

#----------------------------------------------------------------------------

class ParallelDownloader:
    """The downloader can also pseudo-download stuff so that we have a unified interface."""
    def __init__(self, num_workers: int = 4, prefetch: int = 10, num_retries: int = 3, skip_if_exists: bool = True):
        self.downloaders = {} # We assume that all of the downloaders are thread-safe.
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.num_retries = num_retries
        self.skip_if_exists = skip_if_exists
        self.thread_pool: LazyThreadPool = None

    def init_thread_pool(self):
        assert self.thread_pool is None, "Thread pool is already initialized."
        self.thread_pool = LazyThreadPool(
            num_workers=self.num_workers,
            prefetch=self.prefetch,
            num_retries=self.num_retries,
        )

    def __str__(self):
        return (
            f"ParallelDownloader(num_workers={len(self.thread_pool.workers)}, "
            f"prefetch={self.prefetch}, num_retries={self.num_retries}, "
            f"skip_if_exists={self.skip_if_exists}), "
            f'thread_pool={self.thread_pool}'
        )

    def schedule_task(self, key: int | str, source_urls: list[str], destinations: list[str], blocking: bool=False) -> float | None:
        downloading_task = DownloadingTask(
            key=key,
            source_urls=source_urls,
            destinations=destinations,
            timeout=10,
            downloaders=self.downloaders,
            skip_if_exists=self.skip_if_exists,
        )

        for url in source_urls:
            prefix = urllib.parse.urlparse(url).scheme
            if prefix not in self.downloaders or self.downloaders[prefix] is None:
                self.downloaders[prefix] = CloudDownloader.get(url)

        if blocking:
            return run_downloading_task(downloading_task)
        else:
            if self.thread_pool is None:
                self.init_thread_pool()
            self.thread_pool.schedule_task(
                run_downloading_task,
                task_input=downloading_task,
                retries=self.num_retries,
            )

    def reset(self):
        self.thread_pool.reset()

    def wait_completion(self):
        self.thread_pool.wait_completion()

    def stop(self):
        self.thread_pool.stop()

    def shutdown(self):
        if self.thread_pool is not None:
            self.thread_pool.shutdown()

    def _clean_failed_download(self, downloading_task: DownloadingTask):
        """
        Cleans up the failed download by removing the destination files.
        """
        for dst in downloading_task.destinations:
            if os.path.exists(dst):
                logger.debug(f"Removing failed download file: {dst}")
                try:
                    os.remove(dst)
                except Exception as e:
                    logger.error(f"Failed to remove file {dst}: {e}")

    def yield_completed(self) -> Generator:
        for result in self.thread_pool.yield_completed():
            if result["success"]:
                yield (result['task_input'].key, result['task_output'])
            else:
                logger.error(f"Download failed: {result}")
                self._clean_failed_download(result['task_input'])

    def get_num_pending_tasks(self) -> int:
        """
        Returns the number of pending tasks in the thread pool.
        """
        return self.thread_pool.task_queue.qsize()

#----------------------------------------------------------------------------

def run_downloading_task(task: DownloadingTask) -> tuple[float, float]:
    """
    Returns the total size of files in bytes, and the amount of data that has actually been downloaded (not skipped).
    """
    existing_size = 0
    downloaded_size = 0
    for url, dst in zip(task.source_urls, task.destinations):
        prefix = urllib.parse.urlparse(url).scheme
        cur_dst_size = os.path.getsize(dst) if os.path.exists(dst) else 0
        if task.skip_if_exists and cur_dst_size > 0:
            existing_size += cur_dst_size
            continue
        task.downloaders[prefix].download(url, dst, timeout=task.timeout)
        downloaded_size += os.path.getsize(dst)
    return (existing_size + downloaded_size), downloaded_size

#----------------------------------------------------------------------------
