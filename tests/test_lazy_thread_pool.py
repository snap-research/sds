import time

from sds.lazy_thread_pool import LazyThreadPool

def test_single_task_executes():
    pool = LazyThreadPool(num_workers=1)
    results = []

    def simple_task(x):
        results.append(x)

    pool.schedule_task(simple_task, task_input=42)
    pool.wait_completion()

    completed = list(pool.yield_completed())
    assert len(completed) == 1
    assert completed[0]['task_input'] == 42
    assert completed[0]['success'] is True
    assert results == [42]


def test_retry_logic():
    pool = LazyThreadPool(num_workers=1, num_retries=2)
    attempts = []

    def flaky_task(x):
        attempts.append(x)
        if len(attempts) < 3:
            raise ValueError("fail")

    pool.schedule_task(flaky_task, task_input="retry-me")
    pool.wait_completion()
    completed = list(pool.yield_completed())

    assert len(completed) == 1
    assert completed[0]['success'] is True
    assert len(attempts) == 3
    assert completed[0]['num_retries_left'] == 0


def test_retry_exhausted():
    pool = LazyThreadPool(num_workers=1, num_retries=1)

    def always_fail(x):
        raise RuntimeError("bad stuff")

    pool.schedule_task(always_fail, task_input="fail")
    pool.wait_completion()
    completed = list(pool.yield_completed())

    assert len(completed) == 1
    assert completed[0]['success'] is False
    assert "RuntimeError" in completed[0]['error']
    assert completed[0]['num_retries_left'] == 0


def test_pause_resume():
    pool = LazyThreadPool(num_workers=1)
    hits = []

    def slow_task(x):
        hits.append(x)
        time.sleep(0.2)

    # Pause before scheduling to avoid race
    pool.pause()
    pool.schedule_task(slow_task, task_input=123)

    # Sleep enough for the worker thread to try to run, but be blocked by pause
    assert hits == []  # should not have run yet

    pool.resume()

    # Wait a bit for it to start running
    time.sleep(0.3)
    assert hits == [123]

    pool.wait_completion()
    completed = list(pool.yield_completed())
    assert completed[0]['task_input'] == 123
    assert completed[0]['success'] is True


def test_shutdown_cleans_up():
    pool = LazyThreadPool(num_workers=2)

    def task(x):
        return x

    for i in range(10):
        pool.schedule_task(task, task_input=i)

    pool.shutdown()  # no error should be raised


def test_progress_tracking():
    pool = LazyThreadPool(num_workers=1)

    def task(x):
        return x

    # Schedule two tasks: one good, one failing
    pool.schedule_task(task, task_input=1)

    def fail_task(x):
        raise ValueError("boom")

    pool.schedule_task(fail_task, task_input=2)

    pool.wait_completion()
    results = list(pool.yield_completed())

    succeeded = [r for r in results if r["success"]]
    progress = pool.progress()

    assert progress["num_tasks_scheduled"] == 2
    assert progress["num_tasks_completed"] == len(succeeded) == 1


def test_prefetch_limit_enforced():
    prefetch = 5
    num_workers = 3
    total_tasks = 30

    pool = LazyThreadPool(num_workers=num_workers, prefetch=prefetch)

    def short_task(_):
        time.sleep(0.01)  # Simulate some minimal work

    # Schedule all tasks
    for i in range(total_tasks):
        pool.schedule_task(short_task, task_input=i)

    # Allow workers to process tasks
    batches = []
    for _ in range(total_tasks):
        time.sleep(0.03) # Wait until some tasks complete.
        batch = list(pool.yield_completed())
        batches.append(batch)

        assert len(batch) <= prefetch # Assert batch size is no more than prefetch
        assert pool.completed_queue.qsize() == 0 # Internal check: queue should be drained

    # After all, total completed should match total scheduled
    assert pool.progress()["num_tasks_completed"] == total_tasks
    pool.shutdown()

    # Optional: check that all results were received
    all_results = [r for b in batches for r in b]
    assert len(all_results) == total_tasks
    assert all(r["success"] for r in all_results)


def test_threadpool_no_memory_leak():
    import tracemalloc
    import random

    tracemalloc.start()

    def short_task(x) -> int:
        return random.randint(0, 10000)

    def run_pool_cycle():
        pool = LazyThreadPool(num_workers=4, prefetch=10)
        for i in range(100):
            pool.schedule_task(short_task, task_input=i)
        while pool.progress()["num_tasks_completed"] < 100:
            list(pool.yield_completed())
        pool.shutdown()

    # Warm-up run
    run_pool_cycle()
    time.sleep(0.1)
    snapshot1 = tracemalloc.take_snapshot()

    # Repeated runs to detect leak
    for _ in range(10):
        run_pool_cycle()

    time.sleep(0.1)
    snapshot2 = tracemalloc.take_snapshot()

    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_alloc_diff = sum(stat.size_diff for stat in top_stats)

    # Allow small increase due to caching, logger, etc.
    allowed_growth_bytes = 1 * 1024 * 1024  # 1 MB

    assert total_alloc_diff < allowed_growth_bytes, \
        f"Possible memory leak detected: grew by {total_alloc_diff / 1024:.1f} KB"

    tracemalloc.stop()
