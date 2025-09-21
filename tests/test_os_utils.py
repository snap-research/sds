import pytest
from sds.utils.os_utils import path_key

@pytest.mark.parametrize(
    "path,num_parts,drop_ext,expected",
    [
        ("s3://bucket/folder/file.txt", 2, False, "s3-folder-file.txt"),
        ("s3://bucket/folder/", 2, False, "s3-bucket-folder"),
        ("gs://my-bucket/data/2023/part-1.parquet", 2, False, "gs-2023-part-1.parquet"),
        ("gs://my-bucket/data/2023/part-1.parquet", 3, False, "gs-data-2023-part-1.parquet"),
        ("file:///home/user/data.csv", 1, False, "file-data.csv"),
        ("https://example.com/resources/image.png", -1, False, "https-example.com-resources-image.png"),
        ("/var/log/syslog", 2, False, "log-syslog"),
        ("relative/path/to/thing", 2, False, "to-thing"),
        ("s3://bucket/onlyfile", 2, False, "s3-bucket-onlyfile"),
        ("s3://bucket////weird///slashes///x.txt", 2, False, "s3-slashes-x.txt"),

        ("s3://bucket/folder/file.txt", 2, True, "s3-folder-file"),
        ("s3://bucket/folder/", 2, True, "s3-bucket-folder"),
        ("gs://my-bucket/data/2023/part-1.parquet", 2, True, "gs-2023-part-1"),
        ("gs://my-bucket/data/2023/part-1.parquet", 3, True, "gs-data-2023-part-1"),
        ("file:///home/user/data.csv", 1, True, "file-data"),
        ("https://example.com/resources/image.png", -1, True, "https-example.com-resources-image"),
        ("/var/log/syslog", 2, True, "log-syslog"),
        ("relative/path/to/thing", 2, True, "to-thing"),
        ("s3://bucket/onlyfile", 2, True, "s3-bucket-onlyfile"),
        ("s3://bucket////weird///slashes///x.txt", 2, True, "s3-slashes-x"),
    ],
)

def test_path_key_happy_paths(path, num_parts, drop_ext, expected):
    assert path_key(path, num_parts, drop_ext=drop_ext) == expected

def test_uses_all_parts_when_minus_one():
    assert path_key("s3://b/a/b/c.txt", -1) == "s3-b-a-b-c.txt"

def test_assert_on_bad_num_parts():
    with pytest.raises(AssertionError):
        path_key("s3://bucket/file.txt", 0)

def test_empty_path_parts():
    # No scheme, single segment
    assert path_key("single", 1) == "single"
    # Scheme with no netloc/path -> empty => prefix only should result in empty suffix
    assert path_key("file:///superpath.txt", 1) == "file-superpath.txt"
