import os
import urllib.parse
from typing import Generator
from dataclasses import dataclass

from loguru import logger
from streaming.base.storage import CloudDownloader

from sds.lazy_thread_pool import LazyThreadPool


@dataclass
class DownloadingTask:
    key: str
    source_urls: list[str]
    destinations: list[str]
    timeout: int # Timeout in seconds.
    downloaders: dict[str, CloudDownloader]
    skip_if_exists: bool


def run_downloading_task(task: DownloadingTask) -> bool:
    for url, dst in zip(task.source_urls, task.destinations):
        prefix = urllib.parse.urlparse(url).scheme
        if task.skip_if_exists and os.path.exists(dst) and os.path.getsize(dst) > 0:
            logger.debug(f"Skipping download of {url} to {dst} as it already exists.")
            continue
        task.downloaders[prefix].download(url, dst, timeout=task.timeout)
    return os.path.isfile(dst) and os.path.getsize(dst) > 0 # return True if the file was downloaded successfully


class ParallelDownloader:
    """The downloader can also pseudo-download stuff so that we have a unified interface."""
    def __init__(self, data: dict[str, DownloadingTask], num_workers: int = 4, prefetch: int = 10, num_retries: int = 3, skip_if_exists: bool = True):
        self.data = data
        self.thread_pool = LazyThreadPool(
            num_workers=num_workers,
            prefetch=prefetch,
            num_retries=num_retries,
        )
        self.downloaders = {} # We assume that all of the downloaders are thread-safe.
        self.num_retries = num_retries
        self.skip_if_exists = skip_if_exists

    def schedule_task(self, key: str, source_urls: list[str], destinations: list[str]) -> None:
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

        self.thread_pool.schedule_task(
            run_downloading_task,
            task_input=downloading_task,
            retries=self.num_retries,
        )

    def wait_completion(self):
        self.thread_pool.wait_completion()

    def yield_completed(self) -> Generator:
        for result in self.thread_pool.yield_completed():
            if result["success"]:
                yield result['task_input']['key']
            else:
                logger.error(f"Download failed: {result}")
