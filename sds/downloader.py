import os
import urllib.parse
from typing import Generator, Any
from dataclasses import dataclass

from loguru import logger
from streaming.base.storage import CloudDownloader

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
        self.thread_pool = LazyThreadPool(
            num_workers=num_workers,
            prefetch=prefetch,
            num_retries=num_retries,
        )
        self.downloaders = {} # We assume that all of the downloaders are thread-safe.
        self.prefetch = prefetch
        self.num_retries = num_retries
        self.skip_if_exists = skip_if_exists

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
            self.thread_pool.schedule_task(
                run_downloading_task,
                task_input=downloading_task,
                retries=self.num_retries,
            )

    def wait_completion(self):
        self.thread_pool.wait_completion()
        logger.debug(f"All download tasks completed. {self.thread_pool.num_tasks_completed} tasks completed out of {self.thread_pool.num_tasks_scheduled} scheduled.")

    def yield_completed_keys(self) -> Generator:
        for result in self.thread_pool.yield_completed():
            if result["success"]:
                total_downloaded_size = result['task_output']
                yield (result['task_input'].key, total_downloaded_size)
            else:
                logger.error(f"Download failed: {result}")

#----------------------------------------------------------------------------

def run_downloading_task(task: DownloadingTask) -> float:
    """
    Returns the total size of downloaded files in bytes.
    """
    total_size = 0
    for url, dst in zip(task.source_urls, task.destinations):
        prefix = urllib.parse.urlparse(url).scheme
        if task.skip_if_exists and os.path.exists(dst) and os.path.getsize(dst) > 0:
            logger.debug(f"Skipping download of {url} to {dst} as it already exists.")
            continue
        task.downloaders[prefix].download(url, dst, timeout=task.timeout)
        total_size += os.path.getsize(dst) if os.path.exists(dst) else 0
    return total_size

#----------------------------------------------------------------------------
