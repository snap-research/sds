# An example where we load many datasets we have.

import os; os.environ['SDS_LOG_LEVEL'] = 'DEBUG'#; os.environ['AWS_REGION']='us-east-2'  # Set AWS region if needed
from sds.dataloader import MultiStreamDataLoader, StreamOptions

from examples.iter_audio_video_dataset import init_audio_video_dataset
from examples.iter_s3_folder_lora_dataset import init_s3_folder_dataset
from examples.iter_img2img import init_img2img_dataset


def main():
    av_dataset = init_audio_video_dataset()
    s3_folder_dataset = init_s3_folder_dataset()
    img2img_dataset = init_img2img_dataset()

    dataloader = MultiStreamDataLoader(
        datasets=[av_dataset, s3_folder_dataset, img2img_dataset, av_dataset],
        stream_opts=[
            StreamOptions(name='av', ratio=0.33, mixing_group_id=0, batch_size=2),
            StreamOptions(name='s3-folder', ratio=0.33, mixing_group_id=1, batch_size=2),
            StreamOptions(name='av-dummy', ratio=0.0, mixing_group_id=0, batch_size=1), # Dummy stream to test the removal behavior.
            StreamOptions(name='img2img', ratio=0.34, mixing_group_id=2, batch_size=2),
        ],
        shuffle_seed=42,
        schedule='fixed_random_order',
        num_workers=8,
        # num_workers=[2, 2, 3, 2],
        # reweight_workers=False,
    )

    for i, batch in enumerate(dataloader):
        print(f'[iter {i}] Stream: {batch.stream_name}, data_type: {batch.data_type}, num_accum_rounds_left: {batch.num_accum_rounds_left}')
        print(f'[iter {i}] Batch keys', batch.keys())

if __name__ == '__main__':
    main()
