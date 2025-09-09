import os
import shutil
import tempfile
import pytest
from pathlib import Path

from sds.downloader import ParallelDownloader  # Replace with correct import

# Assume S3_URLS_LIST is provided and valid S3 credentials/config are set
S3_URLS_LIST = [f's3://snap-genvid/iskorokhodov/test/{i}.txt' for i in range(100)]

@pytest.fixture
def local_dirs():
    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    yield src, dst
    shutil.rmtree(src)
    shutil.rmtree(dst)

# Local to Local
def test_local_to_local_download(local_dirs):
    src_dir, dst_dir = local_dirs
    filenames = [f"file{i}.txt" for i in range(5)]

    for f in filenames:
        with open(os.path.join(src_dir, f), 'w') as fp:
            fp.write(f"Content {f}")

    downloader = ParallelDownloader(skip_if_exists=False)

    for f in filenames:
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        downloader.schedule_task(f, [src], [dst])

    downloader.wait_completion()

    for f in filenames:
        with open(os.path.join(dst_dir, f)) as fp:
            assert fp.read() == f"Content {f}"

# S3 to Local
@pytest.mark.parametrize("s3_url", S3_URLS_LIST)
def test_s3_to_local(tmp_path, s3_url):
    dst = tmp_path / Path(s3_url).name
    downloader = ParallelDownloader(skip_if_exists=False)
    downloader.schedule_task("s3_test", [s3_url], [str(dst)])
    downloader.wait_completion()
    assert dst.exists()
    assert dst.stat().st_size > 0

# Prefetch Test (verifies staged completion)
def test_prefetch_behavior(local_dirs):
    src_dir, dst_dir = local_dirs
    filenames = [f"f{i}.txt" for i in range(20)]
    for f in filenames:
        with open(os.path.join(src_dir, f), 'w') as fp:
            fp.write("X" * 512)

    downloader = ParallelDownloader(num_workers=2, prefetch=3)
    for f in filenames:
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        downloader.schedule_task(f, [src], [dst])

    completed = []
    for key, _result in downloader.yield_completed():
        completed.append(key)
        assert len(completed) <= 5 or all(os.path.exists(os.path.join(dst_dir, f"{key}.txt")) for key in completed)

# Mixed Local + S3
def test_mixed_sources(local_dirs):
    src_dir, dst_dir = local_dirs
    local_name = "local.txt"
    s3_url = S3_URLS_LIST[0]

    with open(os.path.join(src_dir, local_name), 'w') as fp:
        fp.write("MixedTest")

    local_url = os.path.join(src_dir, local_name)
    dsts = [os.path.join(dst_dir, "from_s3"), os.path.join(dst_dir, "from_local")]

    downloader = ParallelDownloader(skip_if_exists=False)
    downloader.schedule_task("mixed", [s3_url, local_url], dsts)
    downloader.wait_completion()

    for dst in dsts:
        assert os.path.exists(dst)
        assert os.path.getsize(dst) > 0

# skip_if_exists works
def test_skip_if_exists(local_dirs):
    src_dir, dst_dir = local_dirs
    name = "already.txt"
    src = os.path.join(src_dir, name)
    dst = os.path.join(dst_dir, name)

    with open(src, "w") as f:
        f.write("new")
    with open(dst, "w") as f:
        f.write("old")

    downloader = ParallelDownloader(skip_if_exists=True)
    downloader.schedule_task("skip", [src], [dst])
    downloader.wait_completion()

    with open(dst) as f:
        assert f.read() == "old"

# Check that failed downloads do not crash the downloader
@pytest.mark.parametrize("bad_url", [
    "s3://non-existent-bucket-1234567890/fake_file1.txt",
    "s3://this-bucket-does-not-exist-anywhere/fake_file2.jpg",
])
def test_failed_downloads_do_not_crash(tmpdir, bad_url):
    dst_path = os.path.join(str(tmpdir), os.path.basename(bad_url))

    downloader = ParallelDownloader(skip_if_exists=False)
    downloader.schedule_task("bad_s3", [bad_url], [dst_path])
    downloader.wait_completion()

    # Check that the file doesn't exist
    assert not os.path.exists(dst_path)

    # Optionally yield and check failure is reported correctly
    for result_key, _result in downloader.yield_completed():
        # It should NOT yield the key of a failed download
        assert result_key != "bad_s3"
