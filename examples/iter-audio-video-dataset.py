import os; os.environ['SDS_LOG_LEVEL'] = 'DEBUG'
from typing import Callable

import torch
from sds.dataset import StreamingDataset
from sds.transforms import presets


def create_audio_video_transforms_pipeline() -> list[Callable]:
    target_audio_sr = 44100  # Target audio sample rate
    video_transforms = presets.create_standard_joint_video_audio_pipeline(
        video_field='data_url',
        num_frames=1,
        resolution=(256, 256),
        decode_kwargs=dict(
            frame_seek_timeout_sec=10.0,
            framerate=24.0,
            allow_shorter_videos=False,
            random_offset=True,
            approx_frame_seek=True,
            duration_field='duration_s', # Where to find the precomputed duration of the video in seconds.
            framerate_field='fps', # Where to find the precomputed framerate of the video.
            thread_type='NONE',
            frame_timestamps_output_field='frame_timestamps',  # Save the actual timestamps of the decoded frames (in seconds).
        ),
        target_audio_sr=target_audio_sr,
    )

    text_embs_transforms = [
        presets.TextEmbLoaderTransform('short_caption_embedding', num_tokens=512, allow_missing=True),
        presets.TextEmbLoaderTransform('long_caption_embedding', num_tokens=512, allow_missing=True),
        presets.TextEmbSamplingTransform(
            input_text_fields=['short_caption', 'caption'],
            input_text_emb_fields=['short_caption_embedding', 'long_caption_embedding'],
            probabilities=[0.5, 0.5],
            output_text_field='caption',
            output_text_emb_field='caption_embedding',
            allow_missing=True,
            cleanup=True,
        ),
    ]

    return video_transforms + text_embs_transforms


def create_audio_video_dataset():
    dataset = StreamingDataset(
        src='s3://snap-genvid-us-east-2/iskorokhodov/snapvideo_3_datasets/video_audio/67f9ac1f7a6d4ba2a418c2f3ab9731e9/000000000176-last10k.parquet',
        dst='ignore/sv3',
        data_type='video',
        transforms=create_audio_video_transforms_pipeline(),
        columns_to_download=['data_url', 'short_caption_embedding', 'long_caption_embedding'],
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
        infinite_iteration=True,
    )

    return dataset


def main():
    dataset = create_audio_video_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, pin_memory=True, drop_last=True)
    data_iterator = iter(dataloader)

    for epoch in range(1, 2):
        for iteration in range(7):
            batch = next(data_iterator)
            print(f'Processing batch #{iteration + 1} with {len(batch["data_id"])} items.')
            print('batch __sample_key__', batch.get('__sample_key__', None))
            print('Frame timestamps:', batch['frame_timestamps'])

    print('Done')

if __name__ == '__main__':
    main()
