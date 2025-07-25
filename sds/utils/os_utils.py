import os
import re
import gc
import time
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

from beartype import beartype
from loguru import logger
import boto3
from tqdm import tqdm
from sds.utils.download import CloudDownloader

#---------------------------------------------------------------------------

def find_files_in_src(src: str, exts: set[str]=None, **filtering_kwargs) -> list[str]:
    if src.startswith("s3://"):
        return find_files_in_s3_dir(src, exts, **filtering_kwargs)
    elif src.startswith("/"):
        assert os.path.isdir(src), f"Source path {src} is not a valid directory."
        return find_files_in_dir(src, exts, **filtering_kwargs)
    else:
        raise ValueError(f"Unsupported source path: {src}. Must be a local directory or S3 URI.")

def file_ext(f: str) -> str:
    return os.path.splitext(f)[1]

def file_full_ext(f: str) -> str:
    basename = os.path.basename(f)
    parts = basename.split('.')
    return ('.' + '.'.join(parts[1:])) if len(parts) > 1 else ''

def file_key(f: str) -> str:
    """File key is the basename without the full extension."""
    f = f.rstrip('/')
    basename = os.path.basename(f)
    ext = file_full_ext(f)
    return basename[:-len(ext)] if ext else basename

def filter_and_format_files(files, dir_path, exts=None, ignore_regex=None, full_path=True, uri_scheme=""):
    files = [f for f in files if file_ext(f).lower() in exts] if exts else files

    if ignore_regex:
        files = [f for f in files if not re.fullmatch(ignore_regex, f)]

    if full_path:
        files = [os.path.join(dir_path, f) for f in files]
        if uri_scheme:
            files = [f"{uri_scheme}://{f}" for f in files]

    return files

def find_files_in_dir(path: os.PathLike, exts: set[str]=None, **filtering_kwargs) -> list[str]:
    path = os.fspath(path)
    files = [os.path.relpath(os.path.join(r, f), start=path) for r, _, fs in os.walk(path) for f in fs]
    return filter_and_format_files(files, path, exts=exts, uri_scheme="", **filtering_kwargs)


def parallel_download(srcs: list[str], dsts: list[str], skip_if_exists: bool = True, num_workers: int = 16, verbose: bool = False):
    """
    Downloads or copies files from source to destination in parallel using
    a much cleaner `executor.map` approach.

    Args:
        srcs (list[str]): A list of source file paths (local or S3).
        dsts (list[str]): A list of corresponding destination file paths.
        skip_if_exists (bool): If True, skips files where the destination exists.
        num_workers (int): The number of parallel worker threads.
    """
    if len(srcs) != len(dsts):
        raise ValueError("The number of sources must match the number of destinations.")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use executor.map for a concise and clean way to run tasks in parallel.
        # tqdm will automatically create a progress bar for the iterator.
        jobs = executor.map(download_file, srcs, dsts, [skip_if_exists] * len(srcs))
        jobs = tqdm(jobs, total=len(srcs), desc="Downloading files") if verbose else jobs
        _results = list(jobs)

@beartype
def download_file(src: str, dst: str, skip_if_exists: bool, ignore_exceptions: bool = False):
    """
    Wrapper for a single download/copy task. It checks if the file
    exists and calls the downloader.

    Returns True if an operation was performed, False if skipped.
    """
    if skip_if_exists and is_non_empty_file(dst):
        return True

    try:
        CloudDownloader.get(src).direct_download(remote=src, local=dst)
        return True  # Downloaded/Copied
    except Exception as e:
        logger.debug(f"Error processing {src} -> {dst}: {e}")
        if ignore_exceptions:
            return False
        else:
            raise

def is_non_empty_file(path: os.PathLike) -> bool:
    path = os.fspath(path)
    return os.path.isfile(path) and os.path.getsize(path) > 0

def get_file_size(path: str) -> int:
    """
    Get the size of a file in bytes. If the file does not exist, returns 0.
    """
    path = os.fspath(path)
    if os.path.isfile(path):
        return os.path.getsize(path)
    return 0

#---------------------------------------------------------------------------
# Uploading utils.

def upload_file(src: str, dst: str):
    if dst.startswith("s3://"):
        # Upload to S3
        parsed = urlparse(dst)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        s3 = boto3.client("s3")
        s3.upload_file(src, bucket, key)
    elif dst.startswith("/"):
        # Local copy
        dst_dir = os.path.dirname(dst)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        os.replace(src, dst)
    else:
        raise ValueError(f"Unsupported destination path: {dst}. Must be a local path or S3 URI.")

#---------------------------------------------------------------------------
# S3 utils.

def aws_s3_list(s3_path, recursive: bool = True):
    """
    List S3 objects under a given S3 path.

    Args:
        s3_path (str): e.g., 's3://my-bucket/path/to/folder'
        recursive (bool): whether to list recursively or only top-level

    Returns:
        List[str]: list of full S3 paths (starting with 's3://')
    """
    assert s3_path.startswith("s3://")
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    operation_parameters = {"Bucket": bucket, "Prefix": prefix}
    if not recursive:
        operation_parameters["Delimiter"] = "/"

    paths = []
    for page in paginator.paginate(**operation_parameters):
        for obj in page.get("Contents", []):
            paths.append(f"s3://{bucket}/{obj['Key']}")
    return paths


def find_files_in_s3_dir(s3_path: str, exts: set[str], **filtering_kwargs) -> list[str]:
    assert s3_path.startswith("s3://")
    s3 = boto3.client("s3")
    bucket, *parts = s3_path[5:].split("/")
    prefix = "/".join(parts).rstrip("/") + "/"
    files = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/"):
                files.append(obj["Key"][len(prefix):])

    return filter_and_format_files(files, f"{bucket}/{prefix}", exts, uri_scheme="s3", **filtering_kwargs)

#---------------------------------------------------------------------------

class TimeBasedGarbageCollector:
    def __init__(self, interval_seconds: float=None):
        """
        Initialize the garbage collector with a specified interval.

        :param interval_seconds: The interval in seconds between each call to gc.collect(). Disabled by default.
        """
        self.interval_seconds = interval_seconds if interval_seconds is not None else float('inf')
        self.last_gc_time = time.time()

    def maybe_collect(self) -> bool:
        """
        Check if the interval has passed since the last garbage collection.
        If so, run gc.collect() and update the last_gc_time.
        """
        current_time = time.time()
        if current_time - self.last_gc_time > self.interval_seconds:
            gc.collect()
            self.last_gc_time = current_time
            return True
        return False

#---------------------------------------------------------------------------
# Miscellaneous utilities.

def bytes_to_int(bytes_str: int | str) -> int:
    """
    Copy-pasted from: https://github.com/mosaicml/streaming
    Convert human readable byte format to an integer.

    Args:
        bytes_str (Union[int, str]): Value to convert.

    Raises:
        ValueError: Invalid byte suffix.

    Returns:
        int: Integer value of bytes.
    """
    #input is already an int
    if isinstance(bytes_str, int) or isinstance(bytes_str, float):
        return int(bytes_str)

    units = {
        'kb': 1024,
        'mb': 1024**2,
        'gb': 1024**3,
        'tb': 1024**4,
        'pb': 1024**5,
        'eb': 1024**6,
        'zb': 1024**7,
        'yb': 1024**8,
    }
    # Convert a various byte types to an integer
    for suffix in units:
        bytes_str = bytes_str.lower().strip()
        if bytes_str.lower().endswith(suffix):
            try:
                return int(float(bytes_str[0:-len(suffix)]) * units[suffix])
            except ValueError:
                supported_suffix = ['b'] + list(units.keys())
                raise ValueError(''.join([
                    f'Unsupported value/suffix {bytes_str}. Supported suffix are {supported_suffix}.'
                ]))
    else:
        # Convert bytes to an integer
        if bytes_str.endswith('b') and bytes_str[0:-1].isdigit():
            return int(bytes_str[0:-1])
        # Convert string representation of a number to an integer
        elif bytes_str.isdigit():
            return int(bytes_str)
        else:
            supported_suffix = ['b'] + list(units.keys())
            raise ValueError(''.join([
                f'Unsupported value/suffix {bytes_str}. Supported suffix are {supported_suffix}.',
            ]))
#---------------------------------------------------------------------------
