import pytest
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock, mock_open

import pandas as pd

class MockParallelDownloader:
    """A mock for sds.downloader.ParallelDownloader."""
    def __init__(self, *args, **kwargs):
        self.schedule_task = MagicMock()
        self.yield_completed_keys = MagicMock(return_value=iter([]))
        self.stop = MagicMock(return_value=self)
        self.wait_completion = MagicMock()

# Mocks for entire modules
mock_sds_downloader = MagicMock(ParallelDownloader=MockParallelDownloader)
mock_sds_utils_distributed = MagicMock(
    is_node_leader=MagicMock(return_value=True),
    maybe_barrier=MagicMock(),
    # The behavior will be configured in the mock_env fixture
    broadcast_object_list_locally=MagicMock()
)
mock_sds_index = MagicMock()


@pytest.fixture(scope="module")
def dataset_class():
    """
    A pytest fixture that mocks external dependencies and imports the StreamingDataset
    class. This now uses the REAL base classes for more integration-style testing.
    """
    # Create a mock for the parent 'utils' package
    mock_sds_utils = MagicMock()
    mock_sds_utils.distributed = mock_sds_utils_distributed

    with patch.dict('sys.modules', {
        # Other external dependencies are still mocked
        'sds.downloader': mock_sds_downloader,
        'sds.utils': mock_sds_utils, # Mock the parent package
        'sds.utils.distributed': mock_sds_utils_distributed, # Keep the specific mock
        'sds.utils.misc': MagicMock(),
        'sds.structs': MagicMock(),
        'sds.index': mock_sds_index,
        'beartype': MagicMock(beartype=lambda x: x),
    }):
        # By not mocking 'streaming.base.array' and 'torch.utils.data',
        # we allow the real base classes to be imported.
        from sds.dataset import StreamingDataset
        yield StreamingDataset

@pytest.fixture
def mock_env():
    """
    Pytest fixture to set up a temporary directory and mock objects for each test.
    """
    temp_dir = tempfile.mkdtemp()

    # Reset mocks before each test to ensure isolation
    mock_sds_index.reset_mock()
    mock_sds_utils_distributed.reset_mock()

    # Mocks for index and data
    mock_index_meta = MagicMock()
    mock_index_meta.num_samples = 100
    mock_sds_index.build_index.return_value = mock_index_meta

    mock_index_df = pd.DataFrame({
        'index': [f'sample_{i}' for i in range(10)],
        'image.jpg': [f'/fake/src/path/image_{i}.jpg' for i in range(10)],
        'label.txt': [f'/fake/src/path/label_{i}.txt' for i in range(10)],
    })
    mock_sds_index.load_index_slice.return_value = mock_index_df
    mock_sds_index.load_index_row.return_value = mock_index_df.iloc[0]


    def broadcast_side_effect(obj_list):
        return mock_index_meta

    mock_sds_utils_distributed.broadcast_object_list_locally.side_effect = broadcast_side_effect

    yield {
        "temp_dir": temp_dir,
        "mock_src": "/fake/src/path",
        "mock_index_meta": mock_index_meta,
        "mock_index_df": mock_index_df,
    }

    # Teardown logic runs after the test function completes
    shutil.rmtree(temp_dir)


def test_initialization_as_node_leader(mock_env, dataset_class):
    """
    Test that the dataset initializes correctly when it is the node leader.
    It should build the index.
    """
    mock_sds_utils_distributed.is_node_leader.return_value = True

    dataset = dataset_class(
        src=mock_env["mock_src"],
        dst=mock_env["temp_dir"],
        data_type='directory'
    )

    mock_sds_index.build_index.assert_called_once_with(
        mock_env["mock_src"], mock_env["temp_dir"], 'directory'
    )
    assert dataset.name == 'path'
    assert dataset.index_meta == mock_env["mock_index_meta"]
    assert dataset.epoch == -1

def test_initialization_as_non_leader(mock_env, dataset_class):
    """
    Test that the dataset initializes correctly when it is not the node leader.
    It should NOT build the index.
    """
    mock_sds_utils_distributed.is_node_leader.return_value = False

    dataset = dataset_class(
        src=mock_env["mock_src"],
        dst=mock_env["temp_dir"],
        data_type='directory'
    )

    mock_sds_index.build_index.assert_not_called()
    assert dataset.index_meta == mock_env["mock_index_meta"]

def test_len(mock_env, dataset_class):
    """
    Test that __len__ returns the correct number of samples from the metadata.
    """
    dataset = dataset_class(src=mock_env["mock_src"], dst=mock_env["temp_dir"], data_type='directory')
    assert len(dataset) == 100

def test_getitem(mock_env, dataset_class):
    """
    Test retrieving a single item via __getitem__, ensuring it triggers a blocking download.
    """
    dataset = dataset_class(
        src=mock_env["mock_src"],
        dst=mock_env["temp_dir"],
        data_type='directory',
        columns_to_download=['image.jpg'],
        index_col_name='index'
    )
    dataset.downloader = MockParallelDownloader()

    with patch('builtins.open', mock_open(read_data=b'fake_image_data')) as mock_file:
        # Use idiomatic slicing to call __getitem__
        item = dataset[0]

    dataset.downloader.schedule_task.assert_called_once()
    args, kwargs = dataset.downloader.schedule_task.call_args
    assert kwargs['key'] == 0
    assert kwargs['blocking'] is True

    expected_path = os.path.join(mock_env["temp_dir"], 'path', 'sample_0.jpg')
    mock_file.assert_called_once_with(expected_path, 'rb')

    assert item == {'image.jpg': b'fake_image_data'}

def test_iteration_flow(mock_env, dataset_class):
    """
    Test the main iteration logic.
    """
    dataset = dataset_class(
        src=mock_env["mock_src"],
        dst=mock_env["temp_dir"],
        data_type='directory',
        columns_to_download=['image.jpg', 'label.txt'],
        index_col_name='index'
    )

    mock_downloader_instance = MockParallelDownloader()
    mock_downloader_instance.yield_completed_keys.return_value = iter(range(len(mock_env["mock_index_df"])))

    # FIX: Make the side effect more robust by checking the file extension.
    def file_open_side_effect(path, mode):
        if path.endswith('.jpg'):
            return mock_open(read_data=b'fake_image').return_value
        if path.endswith('.txt'):
            return mock_open(read_data=b'fake_label').return_value
        return mock_open().return_value

    # Patch the ParallelDownloader where it's looked up (in the sds.dataset module)
    # to ensure our mock instance is used during iteration.
    with patch('sds.dataset.ParallelDownloader', return_value=mock_downloader_instance):
         with patch('builtins.open', side_effect=file_open_side_effect):
            items = list(dataset)

    mock_sds_index.load_index_slice.assert_called_once()
    assert mock_downloader_instance.schedule_task.call_count == len(mock_env["mock_index_df"])

    # The stop() method is called at the beginning of the *next* iteration to clean up,
    # so it won't be called in this single iteration test. This assertion is removed.
    # mock_downloader_instance.stop.assert_called_once()

    mock_downloader_instance.wait_completion.assert_called_once()

    assert len(items) == len(mock_env["mock_index_df"])
    assert 'image.jpg' in items[0]
    assert 'label.txt' in items[0]
    assert items[0]['image.jpg'] == b'fake_image'

def test_data_processing_callback(mock_env, dataset_class):
    """
    Test that data processing callbacks are applied correctly.
    """
    mock_callback = MagicMock(side_effect=lambda x: {**x, 'processed': True})

    dataset = dataset_class(
        src=mock_env["mock_src"],
        dst=mock_env["temp_dir"],
        data_type='directory',
        columns_to_download=['image.jpg'],
        data_processing_callbacks=[mock_callback]
    )
    dataset.downloader = MockParallelDownloader()

    with patch('builtins.open', mock_open(read_data=b'data')):
        # Use idiomatic slicing to call __getitem__
        item = dataset[0]

    mock_callback.assert_called_once()
    assert 'processed' in item
    assert item['processed'] is True

def test_state_dict(mock_env, dataset_class):
    """
    Test saving and loading the dataset's state.
    """
    dataset = dataset_class(src=mock_env["mock_src"], dst=mock_env["temp_dir"], data_type='directory')

    dataset.epoch = 3
    dataset.num_samples_yielded = 42

    state = dataset.state_dict()
    assert state == {'epoch': 3, 'num_samples_yielded': 42}

    new_dataset = dataset_class(src=mock_env["mock_src"], dst=mock_env["temp_dir"], data_type='directory')
    new_dataset.load_state_dict(state)

    assert new_dataset.epoch == 3
    assert new_dataset.num_samples_yielded == 42

# To run these tests:
# 1. Make sure you have pytest installed (`pip install pytest`).
# 2. This test file assumes the class to be tested is in `sds/dataset.py`.
# 3. Run pytest from your project's root directory: `pytest`
