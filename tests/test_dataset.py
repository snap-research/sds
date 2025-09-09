import os
from unittest.mock import patch, MagicMock, call
import pandas as pd
import pytest
from collections import namedtuple

# Assuming your StreamingDataset class is in `sds.dataset`
from sds.dataset import StreamingDataset

# --- Fixture Definitions ----------------------------------------------------

@pytest.fixture
def mock_env(tmp_path):
    """
    Provides a mock environment with a source path and a temporary destination directory.
    `tmp_path` is a built-in pytest fixture that provides a temporary directory unique to the test function.
    """
    return {
        "mock_src": "/fake/s3/bucket/data",
        "temp_dir": str(tmp_path)
    }

@pytest.fixture
def dataset_class():
    """
    Returns the StreamingDataset class itself.
    """
    return StreamingDataset

# This mock object now includes the 'index_type' attribute to prevent errors,
# although the test logic will bypass the need for it with correct patching.
IndexMeta = namedtuple('IndexMeta', ['num_samples', 'path', 'lazy', 'index_type'])


# --- Test Implementations ---------------------------------------------------

def test_initialization_as_node_leader(mock_env, dataset_class):
    """
    Test that the dataset initializes correctly when it is the node leader.
    It should build the index.
    """
    mock_index_meta = IndexMeta(num_samples=100, path='/mock/index.parquet', lazy=False, index_type='mock')

    # CORRECTED: Patch 'build_index' in the module where it is used ('sds.dataset')
    with patch('sds.utils.distributed.is_node_leader', return_value=True), \
         patch('sds.utils.distributed.init_process_groups'), \
         patch('sds.utils.distributed.maybe_barrier'), \
         patch('sds.utils.distributed.broadcast_object_locally', side_effect=lambda x: x) as mock_broadcast, \
         patch('sds.dataset.build_index', return_value=mock_index_meta) as mock_build_index:

        dataset = dataset_class(
            src=mock_env["mock_src"],
            dst=mock_env["temp_dir"],
            data_type='image',
            columns_to_download=['image_url']
        )

        mock_build_index.assert_called_once()
        mock_broadcast.assert_called_once()
        assert dataset.index_meta == mock_index_meta
        assert len(dataset) == 100


def test_initialization_as_non_leader(mock_env, dataset_class):
    """
    Test that the dataset initializes correctly when it is not the node leader.
    It should NOT build the index.
    """
    mock_index_meta = IndexMeta(num_samples=100, path='/mock/index.parquet', lazy=False, index_type='mock')

    # CORRECTED: Patch 'build_index' in the module where it is used ('sds.dataset')
    with patch('sds.utils.distributed.is_node_leader', return_value=False), \
         patch('sds.utils.distributed.init_process_groups'), \
         patch('sds.utils.distributed.maybe_barrier'), \
         patch('sds.utils.distributed.broadcast_object_locally', return_value=mock_index_meta) as mock_broadcast, \
         patch('sds.dataset.build_index') as mock_build_index:

        dataset = dataset_class(
            src=mock_env["mock_src"],
            dst=mock_env["temp_dir"],
            data_type='image',
            columns_to_download=['image_url']
        )

        mock_build_index.assert_not_called()
        mock_broadcast.assert_called_once()
        assert dataset.index_meta == mock_index_meta
        assert len(dataset) == 100


def test_len(mock_env, dataset_class):
    """
    Test that __len__ returns the correct number of samples from the metadata.
    """
    with patch.object(dataset_class, 'build_index'):
        dataset = dataset_class(src=mock_env["mock_src"], dst=mock_env["temp_dir"], data_type='image', columns_to_download=['image'])
        dataset.index_meta = IndexMeta(num_samples=100, path='/mock/index.parquet', lazy=False, index_type='mock')
        assert len(dataset) == 100


def test_getitem(mock_env, dataset_class):
    """
    Test retrieving a single item via __getitem__, ensuring it triggers a blocking download.
    """
    with patch.object(dataset_class, 'build_index'):
        dataset = dataset_class(src=mock_env["mock_src"], dst=mock_env["temp_dir"], data_type='image', columns_to_download=['image_url'])
        dataset.index_meta = IndexMeta(num_samples=100, path='/mock/index.parquet', lazy=False, index_type='mock')
        dataset.downloader = MagicMock()

        sample_id = 42
        mock_sample_meta = {'index': sample_id, 'image_url': 'http://example.com/image.jpg'}
        mock_df = pd.DataFrame([mock_sample_meta])
        dst_path = os.path.join(dataset.dst, dataset.name, f'{sample_id}-image_url.jpg')

        # CORRECTED: Patch 'read_parquet_slice' using its correct import path
        with patch('sds.dataset.data_utils.read_parquet_slice', return_value=mock_df) as mock_read_slice:
            item = dataset[sample_id]

            mock_read_slice.assert_called_once_with(dataset.index_meta.path, start_offset=sample_id, end_offset=sample_id + 1)
            dataset.downloader.schedule_task.assert_called_once()
            call_args = dataset.downloader.schedule_task.call_args
            assert call_args.kwargs['key'] == sample_id
            assert call_args.kwargs['blocking'] is True
            assert call_args.kwargs['destinations'] == [dst_path]
            assert item['__sample_key__'] == sample_id


def test_iteration_flow(mock_env, dataset_class):
    """
    Test the main iteration logic, including cache eviction.
    """
    with patch.object(dataset_class, 'build_index'), \
         patch('sds.utils.distributed.get_global_worker_info', return_value=(0, 1)):

        dataset = dataset_class(
            src=mock_env["mock_src"],
            dst=mock_env["temp_dir"],
            data_type='image',
            columns_to_download=['image'],
            cache_limit='1kb'
        )
        dataset.index_meta = IndexMeta(num_samples=3, path='/mock/index.parquet', lazy=False, index_type='mock')

        mock_downloader = MagicMock()
        mock_downloader.yield_completed.return_value = [
            ('sample_0', (600, 600)), # Disk usage: 600 bytes
            ('sample_1', (600, 600)), # Disk usage: 1200 bytes > 1024, should evict sample_0
            ('sample_2', (300, 300)), # Disk usage: 600 + 300 = 900 bytes
        ].__iter__()
        mock_downloader.get_num_pending_tasks.return_value = 1000
        dataset.downloader = mock_downloader

        mock_index_df = pd.DataFrame([
            {'index': 'sample_0', 'image': 'url0'},
            {'index': 'sample_1', 'image': 'url1'},
            {'index': 'sample_2', 'image': 'url2'},
        ])

        # CORRECTED: Patch 'load_index_partition' in the module where it's used
        with patch('sds.dataset.load_index_partition', return_value=mock_index_df) as mock_load_partition, \
             patch.object(dataset, '_delete_sample_from_disk') as mock_delete:

            items = list(dataset)

            mock_load_partition.assert_called_once()
            assert len(items) == 3
            mock_delete.assert_called_once()
            deleted_sample_meta = mock_delete.call_args[0][0]
            assert deleted_sample_meta['index'] == 'sample_0'
            assert dataset._worker_disk_usage == 900
            assert list(dataset._stored_sample_keys) == ['sample_1', 'sample_2']


def test_transforms(mock_env, dataset_class):
    """
    Test that transforms are applied correctly.
    """
    def simple_transform(sample):
        sample['transformed'] = True
        return sample

    def generator_transform(sample):
        yield {**sample, 'part': 1}
        yield {**sample, 'part': 2}

    with patch.object(dataset_class, 'build_index'):
        dataset_simple = dataset_class(
            src=mock_env["mock_src"], dst=mock_env["temp_dir"], data_type='image',
            columns_to_download=['image'], transforms=[simple_transform]
        )
        dataset_simple.index_meta = IndexMeta(num_samples=1, path='/mock/index.parquet', lazy=False, index_type='mock')
        # CORRECTED: Patch 'read_parquet_slice' using its correct import path
        with patch('sds.dataset.data_utils.read_parquet_slice', return_value=pd.DataFrame([{'index': 0, 'image': 'url0'}])):
            item = dataset_simple[0]
            assert 'transformed' in item and item['transformed'] is True

    with patch.object(dataset_class, 'build_index'), \
         patch('sds.utils.distributed.get_global_worker_info', return_value=(0, 1)):
        dataset_gen = dataset_class(
            src=mock_env["mock_src"], dst=mock_env["temp_dir"], data_type='image',
            columns_to_download=['image'], transforms=[generator_transform]
        )
        dataset_gen.index_meta = IndexMeta(num_samples=1, path='/mock/index.parquet', lazy=False, index_type='mock')
        dataset_gen.downloader = MagicMock()
        dataset_gen.downloader.yield_completed.return_value = [('sample_0', (100, 100))].__iter__()
        mock_index_df = pd.DataFrame([{'index': 'sample_0', 'image': 'url0'}])
        # CORRECTED: Patch 'load_index_partition' in the module where it's used
        with patch('sds.dataset.load_index_partition', return_value=mock_index_df):
            items = list(dataset_gen)
            assert len(items) == 2
            assert items[0]['part'] == 1 and items[1]['part'] == 2


def test_state_dict(mock_env, dataset_class):
    """
    Test saving and loading the dataset's state.
    """
    with patch.object(dataset_class, 'build_index'):
        dataset1 = dataset_class(src=mock_env["mock_src"], dst=mock_env["temp_dir"], data_type='image', columns_to_download=['image'])
        dataset1.index_meta = IndexMeta(num_samples=100, path='/mock/index.parquet', lazy=False, index_type='mock')
        dataset1.epoch = 5
        dataset1.sample_in_epoch = 42
        state = dataset1.state_dict()
        assert state == {'epoch': 5, 'sample_in_epoch': 42}

        dataset2 = dataset_class(src=mock_env["mock_src"], dst=mock_env["temp_dir"], data_type='image', columns_to_download=['image'])
        dataset2.index_meta = IndexMeta(num_samples=100, path='/mock/index.parquet', lazy=False, index_type='mock')
        dataset2.load_state_dict(state)
        assert dataset2.epoch == 5
        assert dataset2.sample_in_epoch == 42
