import os; os.environ['SDS_LOG_LEVEL'] = 'DEBUG'
from typing import Callable

import torch
import torchvision.transforms.functional as TVF
from sds.dataset import StreamingDataset
from sds.transforms import presets


def create_video_transforms_pipeline(is_target_video: bool=False) -> list[Callable]:
    if is_target_video:
        # In target transforms, we pick up the saved frame timestamps from the source transforms.
        frame_timestamp_kwargs = dict(frame_timestamps_output_field=None, frame_timestamps_input_field='frame_timestamps')
    else:
        # In source transforms, we save the actual timestamps of the decoded frames (in seconds).
        frame_timestamp_kwargs = dict(frame_timestamps_output_field='frame_timestamps', frame_timestamps_input_field=None)
    video_transforms = presets.create_standard_video_pipeline(
        video_field='data_url',
        num_frames=3,
        resolution=(256, 256),
        output_field = 'src_video' if not is_target_video else 'trg_video',
        decode_kwargs=dict(
            frame_seek_timeout_sec=10.0,
            framerate=24.0,
            allow_shorter_videos=False,
            random_offset=True,
            approx_frame_seek=True,
            duration_field='duration_s', # Where to find the precomputed duration of the video in seconds.
            framerate_field='fps', # Where to find the precomputed framerate of the video.
            thread_type='NONE',
            **frame_timestamp_kwargs,
        ),
    )

    return video_transforms


def create_video2video_dataset():
    src_transforms = create_video_transforms_pipeline(is_target_video=False)
    trg_transforms = create_video_transforms_pipeline(is_target_video=True)
    dataset = StreamingDataset(
        src='s3://snap-genvid-us-east-2/iskorokhodov/snapvideo_3_datasets/video_audio/67f9ac1f7a6d4ba2a418c2f3ab9731e9/000000000176-last10k.parquet',
        dst='data/v2v',
        transforms=src_transforms + trg_transforms,
        columns_to_download=['data_url'],
        index_col_name='data_id',
        num_downloading_workers=1,
        shuffle_seed=-1,
        prefetch=100,
        cache_limit='10gb',
        unaligned_worker_index=True,
        allow_missing_columns=True,
        lazy_index_chunk_size=100,
        lazy_index_num_threads=1,
        min_num_pending_tasks_thresh=200,
        # sql_query="SELECT * FROM dataframe WHERE height >= 256 AND width >= 256", # Filter out videos smaller than 256p and without audio
        print_exceptions=True,
        print_traceback=True,
    )

    return dataset


def main():
    dataset = create_video2video_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, pin_memory=True, drop_last=True)
    data_iterator = iter(dataloader)

    for epoch in range(1, 2):
        for iteration in range(7):
            batch = next(data_iterator)
            print(f'Processing batch #{iteration + 1} with {len(batch["data_id"])} items.')
            print('batch __sample_key__', batch.get('__sample_key__', None))
            print(f'Src shape: {batch["src_video"].shape}, Trg shape: {batch["trg_video"].shape}')
            # Saving the second frame from src/trg videos side-by-side for visual inspection (they should be equivalent)
            save_dir = 'data/v2v/samples-vis'
            os.makedirs(save_dir, exist_ok=True)
            for i in range(len(batch['data_id'])):
                src_frame = batch['src_video'][i, 1]  # (3, H, W)
                trg_frame = batch['trg_video'][i, 1]  # (3, H, W)
                combined = torch.cat([src_frame, trg_frame], dim=2)  # (3, H, 2*W)
                combined_pil = TVF.to_pil_image(combined)
                combined_pil.save(os.path.join(save_dir, f'epoch{epoch:03d}_iter{iteration:03d}_item{i:03d}.png'))


    print('Done')

if __name__ == '__main__':
    main()
