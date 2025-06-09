"""
A general-purpose thread-pool logic class which can execute tasks with prefetching.
"""

import time
import queue
import threading

from loguru import logger

#----------------------------------------------------------------------------

class Worker(threading.Thread):
    def __init__(
            self,
            worker_id: int,
            task_queue,
            completed_queue,
            pause_event,
            stop_event,
        ):
        super().__init__()

        self.worker_id = worker_id
        self.task_queue = task_queue
        self.completed_queue = completed_queue
        self.pause_event = pause_event
        self.stop_event = stop_event
        self.daemon = True
        self.start()

    def run(self):
        is_retrying_prev_task = False

        while not self.stop_event.is_set():
            self.pause_event.wait()
            if not is_retrying_prev_task:
                try:
                    task_fn, task_input, num_retries_left = self.task_queue.get(timeout=1)
                except queue.Empty:
                    continue

            result = {
                "task_input": task_input,
                "task_output": None,
                "success": True,
                "error": None,
                "worker_id": self.worker_id,
                "num_retries_left": -1,
            }

            try:
                result['task_output'] = task_fn(task_input)
            except Exception as e:
                if num_retries_left > 0:
                    logger.debug(f"[Worker {self.worker_id}] Retrying task {task_input} due to error: {repr(e)}")
                    is_retrying_prev_task = True
                    num_retries_left -= 1
                    continue
                else:
                    result["success"] = False
                    result["error"] = repr(e)

            is_retrying_prev_task = False
            result['num_retries_left'] = num_retries_left
            self.completed_queue.put(result)
            self.task_queue.task_done()

#----------------------------------------------------------------------------

class LazyThreadPool:
    """A thread pool which executes tasks only up to prefetching limit (hence, lazy)."""
    def __init__(self, num_workers=4, prefetch=10, num_retries=1):
        self.task_queue = queue.Queue()
        self.completed_queue = queue.Queue(maxsize=prefetch)
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.stop_event = threading.Event()
        self.num_retries = num_retries
        worker_args = (self.task_queue, self.completed_queue, self.pause_event, self.stop_event)
        self.workers = [Worker(wid, *worker_args) for wid in range(num_workers)]
        self.num_tasks_scheduled = 0
        self.num_tasks_completed = 0

    def schedule_task(self, task_fn, task_input=None, retries=None):
        if retries is None:
            retries = self.num_retries
        self.task_queue.put((task_fn, task_input, retries))
        self.num_tasks_scheduled += 1

    def pause(self):
        self.pause_event.clear()

    def resume(self):
        self.pause_event.set()

    def shutdown(self):
        self.stop_event.set()
        for worker in self.workers:
            worker.join()

    def __del__(self):
        self.shutdown()

    def yield_completed(self) -> iter:
        while not self.completed_queue.empty():
            task_result = self.completed_queue.get()
            self.num_tasks_completed += 1 if task_result['success'] else 0
            yield task_result

    def progress(self) -> dict[str, int]:
        return dict(num_tasks_completed=self.num_tasks_completed, num_tasks_scheduled=self.num_tasks_scheduled)

    def wait_completion(self):
        self.task_queue.join()

#----------------------------------------------------------------------------
