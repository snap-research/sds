import unittest
import os
import tempfile
import pandas as pd
import pyarrow.parquet as pq
from unittest.mock import patch, MagicMock

# Assume the code to be tested is in a file named 'your_module.py'
# If it's in the same file, you can import directly.
from sds.index import (
    DataSampleType,
    IndexMetaData,
    IndexType,
    RAW_INDEX_FILES_DIR,
    INDEX_FILE_NAME,
    build_index_from_files_list,
    build_index_from_index_file,
    load_index_slice,
)

# Mocking the dependencies since they are not provided
class MockDistUtils:
    def is_local_main_process(self):
        return True

    def get_rank(self):
        return 0

    def get_node_rank(self):
        return 0

    def get_num_nodes(self):
        return 1

class MockDataUtils:
    def save_polars_parquet(self, df, path):
        df.to_parquet(path)

# You would place the original code in a file, let's call it 'sds_index_builder.py'
# For this example, I'll assume the classes and functions are available in the global scope.
# To make the original code runnable, we need to define the mocked modules
dist_utils = MockDistUtils()
data_utils = MockDataUtils()
# Let's also create dummy versions of the other missing modules for the sake of making the code executable
class MockCloudDownloader:
    @staticmethod
    def get(src):
        class Downloader:
            def download(self, src, dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, 'w') as f:
                    f.write(f"path1.csv\npath2.csv")
        return Downloader()

torch = MagicMock()
pl = MagicMock()
Image = MagicMock()
Image.registered_extensions = {'.jpg', '.png'}


# The provided code starts here...
import random
from dataclasses import dataclass
from enum import Enum
import pyarrow as pa

# (Paste the user's provided code here)
# ... (The user's code from the prompt) ...
# For brevity, I will omit pasting the entire code block again.
# Assuming the user's code is now present in this scope.
# I will need to redefine the mocked dependencies inside the module's scope if they are imported there.
# To do this properly, we'd use patching.

class TestFileFunctions(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory for tests."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.dst_dir = self.test_dir.name
        # Create a subdirectory for raw files as the functions expect it
        os.makedirs(os.path.join(self.dst_dir, RAW_INDEX_FILES_DIR), exist_ok=True)

    def tearDown(self):
        """Clean up the temporary directory."""
        self.test_dir.cleanup()

    def test_data_sample_type_enum(self):
        """Test the DataSampleType enum."""
        self.assertEqual(str(DataSampleType.IMAGE), 'IMAGE')
        self.assertEqual(DataSampleType.from_str('VIDEO'), DataSampleType.VIDEO)
        with self.assertRaises(KeyError):
            DataSampleType.from_str('UNKNOWN')

    def test_build_index_from_files_list(self):
        """Test building an index from a list of files."""
        # Create dummy files
        dummy_files = ['img1.jpg', 'img1.txt', 'img2.png', 'img2.txt']
        file_paths = [os.path.join(self.dst_dir, f) for f in dummy_files]
        for f_path in file_paths:
            with open(f_path, 'w') as f:
                f.write("dummy")

        index_meta = build_index_from_files_list(file_paths, self.dst_dir, DataSampleType.IMAGE)

        # Assertions
        self.assertIsInstance(index_meta, IndexMetaData)
        self.assertEqual(index_meta.num_samples, 2)
        self.assertEqual(index_meta.index_type, IndexType.INTER_NODE)

        # Check the created parquet file
        output_parquet = os.path.join(self.dst_dir, INDEX_FILE_NAME)
        self.assertTrue(os.path.exists(output_parquet))

        df = pd.read_parquet(output_parquet)
        self.assertEqual(len(df), 2)
        self.assertIn('index', df.columns)
        self.assertIn('jpg', df.columns)
        self.assertIn('txt', df.columns)
        self.assertIn('img1', df['index'].values)

    def test_build_index_from_index_file(self):
        """Test building an index from a single CSV index file."""
        # Create a dummy csv file
        src_csv_path = os.path.join(self.dst_dir, 'test_index.csv')
        dummy_df = pd.DataFrame({'image_path': ['/path/to/img1.jpg', '/path/to/img2.jpg']})
        dummy_df.to_csv(src_csv_path, index=False)

        index_meta = build_index_from_index_file(src_csv_path, self.dst_dir)

        # Assertions
        self.assertEqual(index_meta.num_samples, 2)
        self.assertEqual(index_meta.index_type, IndexType.INTER_NODE)

        output_parquet = os.path.join(self.dst_dir, INDEX_FILE_NAME)
        self.assertTrue(os.path.exists(output_parquet))

        df = pd.read_parquet(output_parquet)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ['image_path'])

    def test_load_index_slice(self):
        """Test loading a slice of the index."""
        # Create a dummy parquet file
        num_samples = 100
        dummy_df = pd.DataFrame({'data': range(num_samples)})
        index_path = os.path.join(self.dst_dir, 'full_index.parquet')
        dummy_df.to_parquet(index_path)
        index_meta = IndexMetaData(num_samples=num_samples, path=index_path, index_type=IndexType.INTER_NODE)

        num_ranks = 4

        # Mock pl.scan_parquet for this specific test
        with patch('polars.scan_parquet') as mock_scan:
            # Re-implement a simplified version of the slicing logic using pandas
            def scan_side_effect(path):
                class MockScanner:
                    def __init__(self, df):
                        self._df = df
                    def slice(self, offset, length):
                        class MockSliced:
                            def __init__(self, sliced_df):
                                self._sliced_df = sliced_df
                            def collect(self):
                                return self # simplified chain
                            def to_pandas(self):
                                return self._sliced_df
                        return MockSliced(self._df.iloc[offset:offset+length])

                # We need to read the real file to test the slicing logic
                real_df = pd.read_parquet(path)
                return MockScanner(real_df)

            mock_scan.side_effect = scan_side_effect

            # Rank 0
            rank_0_df = load_index_slice(index_meta, rank=0, num_ranks=num_ranks, num_nodes=1)
            self.assertEqual(len(rank_0_df), 25)
            self.assertEqual(rank_0_df['data'].iloc[0], 0)

            # Rank 2
            rank_2_df = load_index_slice(index_meta, rank=2, num_ranks=num_ranks, num_nodes=1)
            self.assertEqual(len(rank_2_df), 25)
            self.assertEqual(rank_2_df['data'].iloc[0], 50)

            # Last rank
            rank_3_df = load_index_slice(index_meta, rank=3, num_ranks=num_ranks, num_nodes=1)
            self.assertEqual(len(rank_3_df), 25)
            self.assertEqual(rank_3_df['data'].iloc[0], 75)


if __name__ == '__main__':
    # Before running, make sure to define/import the functions and classes under test.
    # To run this, you would save the original code as, e.g., 'sds_index_builder.py'
    # and this test script as 'test_sds_index_builder.py'.
    # Then run 'python -m unittest test_sds_index_builder.py'

    # For this self-contained example, we'll patch the missing modules
    # This setup is complex because the original code has many external dependencies.
    # A real-world setup would involve a proper project structure.
    with patch.dict('sys.modules', {
        'torch': torch,
        'torch.distributed': MagicMock(),
        'pyarrow': pa,
        'pyarrow.parquet': pq,
        'pandas': pd,
        'polars': pl,
        'streaming.base.storage': MagicMock(CloudDownloader=MockCloudDownloader),
        'PIL': MagicMock(Image=Image),
        'sds': MagicMock(),
        'sds.utils': MagicMock(),
        'sds.utils.distributed': dist_utils,
        'sds.utils.data_utils': data_utils,
    }):
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
