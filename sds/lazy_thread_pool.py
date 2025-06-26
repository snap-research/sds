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
            task_wait_timeout: float = 0.1,
        ):
        super().__init__()

        self.worker_id = worker_id
        self.task_queue = task_queue
        self.completed_queue = completed_queue
        self.pause_event = pause_event
        self.stop_event = stop_event
        self.task_wait_timeout = task_wait_timeout
        self.daemon = True
        self.start()

    def run(self):
        is_retrying_prev_task = False

        while not self.stop_event.is_set():
            self.pause_event.wait()
            if not is_retrying_prev_task:
                try:
                    task_fn, task_input, num_retries_left = self.task_queue.get(timeout=self.task_wait_timeout)
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

        logger.debug(f"[Worker {self.worker_id}] Stopping worker thread.")

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

    def __str__(self):
        return (
            f"LazyThreadPool(num_workers={len(self.workers)}, "
            f"prefetch={self.completed_queue.maxsize}, "
            f"num_retries={self.num_retries}, "
            f"num_tasks_scheduled={self.num_tasks_scheduled}, "
            f"num_tasks_completed={self.num_tasks_completed})"
        )

    def schedule_task(self, task_fn, task_input=None, retries=None):
        if retries is None:
            retries = self.num_retries
        self.task_queue.put((task_fn, task_input, retries))
        self.num_tasks_scheduled += 1

    def pause(self):
        self.pause_event.clear()

    def resume(self):
        self.pause_event.set()

    def stop(self):
        self.stop_event.set()
        self.pause_event.set()

    def shutdown(self):
        self.stop()
        for worker in self.workers:
            worker.join(timeout=1)

    def clear_pending_tasks(self) -> int:
        """Clears all tasks from the queue that have not yet started executing."""
        num_cleared = 0
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait() # Use get_nowait() to avoid blocking.
                self.task_queue.task_done() # Must call task_done() to keep the queue's internal counter correct for wait_completion().
                num_cleared += 1
            except queue.Empty:
                # This handles a rare race condition where the queue becomes
                # empty between the .empty() check and the .get_nowait() call.
                break

        if num_cleared > 0:
            # Adjust the scheduled count so that yield_completed() does not wait for tasks that will never arrive.
            self.num_tasks_scheduled -= num_cleared
            logger.info(f"Cleared {num_cleared} pending tasks from the queue.")

        return num_cleared

    def drain_completed_tasks(self) -> int:
        while not self.completed_queue.empty():
            try:
                self.completed_queue.get_nowait()
            except queue.Empty:
                break

    def reset(self):
        self.num_tasks_scheduled = 0
        self.num_tasks_completed = 0
        self.clear_pending_tasks()
        self.drain_completed_tasks()

    def yield_completed(self) -> iter:
        """
        Yields completed task results as they become available.

        This method will block and wait for results to appear in the completed_queue
        and will yield them one by one until all scheduled tasks have been accounted for.
        """
        num_yielded = 0
        while num_yielded < self.num_tasks_scheduled:
            # get() is a blocking call. It will wait here indefinitely until a
            # worker puts a result in the queue. It does NOT depend on timing.
            task_result = self.completed_queue.get()
            num_yielded += 1
            self.num_tasks_completed += 1 if task_result['success'] else 0

            yield task_result

    def progress(self) -> dict[str, int]:
        return dict(num_tasks_completed=self.num_tasks_completed, num_tasks_scheduled=self.num_tasks_scheduled)

    def wait_completion(self):
        self.task_queue.join()

#----------------------------------------------------------------------------
