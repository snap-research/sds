import os; os.environ['SDS_LOG_LEVEL'] = 'DEBUG'#; os.environ['AWS_REGION']='us-east-2'  # Set AWS region if needed
import time
from sds.dataset import StreamingDataset
from sds.transforms import presets
from sds.dataloader import MultiStreamDataLoader, StreamOptions

def main():
    now = time.time()
    video_transforms = presets.create_standard_video_pipeline(
        video_field='mp4',
        num_frames=16,
        resolution=(256, 256),
        decode_kwargs=dict(
            frame_seek_timeout_sec=20.0,
            framerate=12.0,
            allow_shorter_videos=False,
            random_offset=True,
        ),
    )

    # text_embs_transforms = [
    #     presets.TextEmbLoaderTransform('short_caption_embedding', num_tokens=512, allow_missing=True),
    #     presets.TextEmbLoaderTransform('long_caption_embedding', num_tokens=512, allow_missing=True),
    #     presets.TextEmbSamplingTransform(
    #         input_text_fields=['summary_text.json'],
    #         input_text_emb_fields=['summary_text_embeddings.pkl'],
    #         probabilities=[1.0],
    #         output_text_field='caption',
    #         output_text_emb_field='caption_embedding',
    #         allow_missing=True,
    #         cleanup=True,
    #     ),
    # ]
    text_embs_transforms = []

    # The original init can take a bit of time, since it downloads the index metadata (400Mb in the case below).
    dataset = StreamingDataset(
        src='s3://snap-genvid-us-east-2/datasets/snapvideo_lora_webtool/250905_hair_growth_rerun_pl_37b1ef33/dataset_tmp/',
        dst='ignore/sv3-lora',
        data_type='video',
        transforms=video_transforms + text_embs_transforms,
        columns_to_download=['mp4', 'summary_text.json', 'summary_text_embeddings.pkl', 'vidinfo.json'],
        index_col_name='index',
        num_downloading_workers=1,
        shuffle_seed=-1,
        prefetch=100,
        cache_limit='10gb',
        allow_missing_columns=True,
        min_num_pending_tasks_thresh=200,
        print_exceptions=True,
        print_traceback=True,
        unaligned_worker_index=True,
    )
    print(f'Init took {time.time() - now:.2f} seconds')

    for epoch in range(1, 2):
        now = time.time()
        stream_opts = StreamOptions.init_group([dict(name='stream1', batch_gpu=2, num_accum_rounds=2, is_main=True, ratio=0.5, mixing_group_id=0)], mixing_strategy='custom')
        dataloader = MultiStreamDataLoader(
            datasets=[dataset],
            stream_opts=stream_opts,
            schedule='fixed_random_order',
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        data_iterator = iter(dataloader)
        for i in range(7):
            batch = next(data_iterator)
            print(f'Processing batch #{i + 1} with {batch["video"].shape} shape. Is memory pinned? {batch["video"].is_pinned()}')

    print('Done')

if __name__ == '__main__':
    main()
