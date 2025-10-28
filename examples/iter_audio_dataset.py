import os; os.environ['SDS_LOG_LEVEL'] = 'DEBUG'
from typing import Callable

import torch
from sds.dataset import StreamingDataset
import torchaudio
from sds.transforms import presets


def build_audio_transforms_pipeline() -> list[Callable]:
    target_audio_sr = 44100  # Target audio sample rate
    audio_transforms = presets.create_standard_audio_pipeline(
        audio_field='data_url',
        duration=2.0,
        mono_audio=True,
        audio_decoding_kwargs=dict(
            # random_offset=False, max_mean_offset=True, # Sampling the loudest audio segment.
            # random_offset=True, max_mean_offset=False,
            # clip_offset_strategy='max_center',
            clip_offset_strategy='max_random_loud',
            allow_shorter_audio=False,
        ),
        audio_resampling_kwargs=dict(

        ),
        target_audio_sr=target_audio_sr,
    )

    return audio_transforms


def init_audio_dataset():
    dataset = StreamingDataset(
        # src='s3://snap-genvid-us-east-2/datasets/sds-index-files/iskorokhodov/av/bbc-sounds.parquet',
        src='s3://snap-genvid-us-east-2/datasets/sds-index-files/iskorokhodov/av/bbc-sounds-val.parquet',
        dst='data/bbc-sounds',
        transforms=build_audio_transforms_pipeline(),
        columns_to_download=['data_url'],
        index_col_name='data_id',
        num_downloading_workers=3,
        shuffle_seed=42,
        prefetch=100,
        cache_limit='100gb',
        allow_missing_columns=False,
        lazy_index_chunk_size=None, # The "BBC sounds" dataset is too small, we don't need lazy indexing.
        min_num_pending_tasks_thresh=200,
        # sql_query="SELECT * FROM dataframe WHERE height >= 256 AND width >= 256", # Filter out videos smaller than 256p and without audio
        print_exceptions=True,
        print_traceback=True,
        unaligned_worker_index=False,
        infinite_iteration=False,
    )

    return dataset


def main():
    dataset = init_audio_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=4, pin_memory=True, drop_last=True)
    data_iterator = iter(dataloader)

    for epoch in range(1, 2):
        for iteration in range(7):
            batch = next(data_iterator)
            print(f'Processing batch #{iteration + 1} with {len(batch["data_id"])} items.')
            print('batch __sample_key__', batch.get('__sample_key__', None), batch['audio'].shape)
            for sample_idx in range(len(batch['data_id'])):
                print(f'  Sample #{sample_idx + 1}: id={batch["data_id"][sample_idx]}, audio shape={batch["audio"][sample_idx].shape}, min={batch["audio"][sample_idx].min().item():.5f}, max={batch["audio"][sample_idx].max().item():.5f}, mean={batch["audio"][sample_idx].mean().item():.5f}, std={batch["audio"][sample_idx].std().item():.5f}')
                # Save the audio to a file for verification
                os.makedirs('data/bbc-sounds/real-samples', exist_ok=True)
                torchaudio.save(f'data/bbc-sounds/real-samples/output_audio_{epoch}_{iteration}_{batch["__sample_key__"][sample_idx]}.wav', batch['audio'][sample_idx], sample_rate=44100)

    print('Done')

if __name__ == '__main__':
    main()
