import sys  # Adjust the path to include the parent directory
import threading
import os; os.environ['SDS_LOG_LEVEL'] = 'DEBUG'#; os.environ['AWS_REGION']='us-east-2'  # Set AWS region if needed
import torch
import time
import numpy as np
import psutil

def print_thread_info():
    """Gets thread counts from both Python's and the OS's perspective."""
    python_thread_count = threading.active_count()

    # Get a handle to the current process
    current_process = psutil.Process(os.getpid())
    os_thread_count = current_process.num_threads()

    print(f"Python's view (threading.active_count): {python_thread_count} threads")
    print(f"OS's view (psutil.num_threads)    : {os_thread_count} threads")
    print("-" * 30)

print_thread_info()

from sds.dataset import StreamingDataset
from sds.transforms import presets
from sds.dataloader import MultiStreamDataLoader, StreamOptions


print_thread_info()

def main():
    now = time.time()
    image_transforms = presets.create_standard_image_pipeline(
        image_field='data_url',
        resolution=(256, 256),
    )

    print_thread_info()

    # The original init can take a bit of time, since it downloads the index metadata (400Mb in the case below).
    dataset = StreamingDataset(
        # src='s3://snap-datastream/SnapGenv4/sds_preprocessing/akag/*.parquet',
        src='s3://snap-datastream/SnapGenv4/sds_preprocessing/akag/v0_nsfwfiltered_treasure_0p66_below_images.parquet',
        dst='ignore/sv3',
        data_type='image',
        transforms=image_transforms,
        columns_to_download=['data_url'],
        index_col_name='data_id',
        num_downloading_workers=1,
        shuffle_seed=-1,
        # max_size=64,
        prefetch=100,
        # cache_limit='10gb',
        cache_limit=0,
        allow_missing_columns=True,
        # lazy_index_chunk_size=100,
        # lazy_index_num_threads=1,
        min_num_pending_tasks_thresh=200,
        # sql_query="SELECT * FROM dataframe WHERE height >= 256 AND width >= 256", # Filter out videos smaller than 256p and without audio
        print_exceptions=True,
        print_traceback=True,
        max_size=5000,
    )
    print(f'Init took {time.time() - now:.2f} seconds')

    for epoch in range(1, 2):
        now = time.time()
        # data_iterator = iter(torch.utils.data.DataLoader(dataset, batch_size=13, num_workers=0, pin_memory=True, drop_last=True))
        stream_opts = StreamOptions.init_group([
            dict(name='stream1', batch_gpu=8, num_accum_rounds=2, is_main=True, ratio=0.5, mixing_group_id=0),
            # dict(name='stream2', batch_gpu=4, num_accum_rounds=4, is_main=True, ratio=0.5, mixing_group_id=2),
        ], mixing_strategy='custom')
        dataloader = MultiStreamDataLoader(
            # datasets=[dataset, dataset],
            datasets=[dataset],
            stream_opts=stream_opts,
            schedule='fixed_random_order',
            num_workers=2,
            # num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        data_iterator = iter(dataloader)
        print(f'Creating DataLoader took {time.time() - now:.2f} seconds')
        for i in range(7):
            print_thread_info()
        # for i in range(np.random.randint(1, 2)):
            batch = next(data_iterator)
            print(f'Processing batch #{i + 1} with {len(batch["data_id"])} items. Is memory pinned? {batch["image"].is_pinned()}')
            # print(f'iter {i} done')
            # import torchaudio
            # for i in range(len(batch['audio'])):
            #     torchaudio.save(f"ignore/sva/000000000176-last10k/{batch['data_id'][i]}.wav", batch['audio'][i], sample_rate=target_audio_sr)

    print('Done')

if __name__ == '__main__':
    main()
