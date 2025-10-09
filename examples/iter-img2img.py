import os
from typing import Callable
os.environ['AWS_REGION'] = 'us-west-2' # Make sure that AWS_REGION is set to the region of your S3 bucket.
os.environ['SDS_LOG_LEVEL'] = 'DEBUG' # Set the log level to DEBUG to see more logs.

import numpy as np
import torch
from PIL import Image
from sds.dataset import StreamingDataset
from sds.transforms import presets


def build_transforms() -> list[Callable]:
    # SDS contains some standard transforms for common use cases. Let's use the one for images.
    src_image_transforms = presets.create_standard_image_pipeline(image_field='src_image_url', output_field='src_image', resolution=(256, 256))
    trg_image_transforms = presets.create_standard_image_pipeline(image_field='trg_image_url', output_field='trg_image', resolution=(256, 256))

    # Let's only keep the necessary fields in the final output, since there is a lot of unused stuff in the index file.
    # (This unused stuff can break pytorch's default collate function.)
    fields_to_keep = ['src_image', 'trg_image', 'src_caption', 'trg_caption', 'src_media_id', 'trg_media_id', '__sample_key__']
    sample_cleaning_transform = presets.EnsureFieldsTransform(fields_whitelist=fields_to_keep, drop_others=True)

    text_loading_transforms = [
        presets.LoadFromDiskTransform(fields_to_load=['src_caption_url', 'trg_caption_url'], mode='r'),
        presets.RenameFieldsTransform(old_to_new_mapping={'src_caption_url': 'src_caption', 'trg_caption_url': 'trg_caption'}),
    ]

    return src_image_transforms + trg_image_transforms + text_loading_transforms + [HorizontalFlipTransform('src_image'), sample_cleaning_transform]


class HorizontalFlipTransform:
    # A simple horizontal flip transform just for illustration purposes
    # Transform is just a class with a __call__ method which takes a sample dict and returns a sample dict.
    def __init__(self, image_field: str):
        self.image_field = image_field

    def __call__(self, sample: dict) -> dict:
        assert self.image_field in sample, f'Image field is missing in the sample: {sample.keys()}'
        assert isinstance(sample[self.image_field], torch.Tensor), f'Image is not a torch.Tensor, but {type(sample["image"])}'
        assert sample[self.image_field].ndim == 3, f'Wrong image shape: {sample["image"].shape}'
        assert sample[self.image_field].shape[0] in [1, 3], f'Wrong image shape: {sample["image"].shape}'
        sample[self.image_field] = torch.flip(sample[self.image_field], dims=[2]) if torch.rand(1) < 0.5 else sample[self.image_field] # [c, h, w]
        return sample


class SampleRandomImagePairTransform:
    def __call__(self, sample: dict) -> dict:
        media_items = sample['media_items']
        assert len(media_items) >= 2, f'Not enough media items: {len(media_items)}. Sample face_series_key: {sample["face_series_key"]}'
        np.random.shuffle(media_items)
        image_src, image_trg = media_items[:2]

        sample['src_image_url'] = image_src['personalization_image_path']
        sample['src_caption_url'] = image_src['personalization_caption_path']
        sample['src_media_id'] = image_src['media_id']
        sample['trg_image_url'] = image_trg['personalization_image_path']
        sample['trg_caption_url'] = image_trg['personalization_caption_path']
        sample['trg_media_id'] = image_trg['media_id']

        return sample


def main():
    dataset = StreamingDataset(
        src='s3://snap-genvid/datasets/sds-index-files/gss-personalization.parquet',
        dst='/tmp/where/to/download',      # Where to download the samples.
        transforms=build_transforms(),     # A list of transforms to apply.
        predownload_transforms=[SampleRandomImagePairTransform()], # Transforms to apply before downloading. Used to sample which columns to download.

        # Which columns to download from the index. Their values should be URLs/paths.
        # Once downloaded, a sample dict will be constructed, with the column names pointing to the local paths.
        # After that, all the transforms will be applied to the sample dict.
        columns_to_download=['src_image_url', 'trg_image_url', 'src_caption_url', 'trg_caption_url'],

        index_col_name='face_series_key',  # Name of the column to use as the index (should have unique values for samples).
        num_downloading_workers=5,         # How many parallel downloading workers to use.
        prefetch=100,                      # How many samples to pre-download ahead of time.
        cache_limit='10gb',                # How much disk space to use for caching.
        allow_missing_columns=False,       # Should we ignore or not ignore samples with some missing `columns_to_download`?

        # Some configuration for the `lazy index`. Sometimes, the index file is huge, so it's better
        # not to load it all at once, but rather fetch in chunks.
        lazy_index_chunk_size=5000,         # Chunk size to fetch.
        lazy_index_num_threads=2,           # In how many threads to fetch the index chunks.
        lazy_index_prefetch_factor=3,       # How many chunks to prefetch ahead of time.
        min_num_pending_tasks_thresh=200,   # How many downloading samples should there be pending, before we start scheduling for sample downloading the next index chunk.

        # Shuffle seed to use. If None, no shuffling is performed.
        # If -1, then a random seed will be created on the first rank and distributed across all the ranks.
        shuffle_seed=123,

        # Let's filter out the images without any related images.
        # The SQL query is quite arbitrary and applied on each index chunk before scheduling downloading tasks.
        sql_query=None,

        # You can limit the dataset size artifically for debugging purposes (i.e. trying to overfit).
        # Note: to reduce the dataset size below the batch_gpu size, you need to also set unaligned_worker_index=True and infinite_iteration=True.
        max_size=50_000,
        # unaligned_worker_index: bool = False, # Shall each worker iterate over the global dataset independently? Bad design, but helpful for tiny datasets.
        # infinite_iteration: bool = False, # If True, the dataset would be iterated infinitely. Useful when you for some reason have batch_size > dataset_size and drop_last=True.

        # Enable these flags for debugging purposes. Otherwise, the exceptions will be silenced.
        print_exceptions=True,
        print_traceback=True,
    )

    # Note: we have (slow) random access!
    sample = dataset[12345]
    img_pair = torch.cat([sample['src_image'], sample['trg_image']], dim=2) # [c, h, w*2]
    img_pair = img_pair.permute(1, 2, 0).cpu().numpy().astype('uint8') # [h, w, c]
    Image.fromarray(img_pair).save('/tmp/where/to/download/debug-sample-12345.png')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=2, pin_memory=True, drop_last=True)
    data_iterator = iter(dataloader)

    for epoch in range(1, 2):
        for i in range(7):
            batch = next(data_iterator)
            print(f'Processing batch #{i + 1} with {len(batch["__sample_key__"])} items.')
            for i, src_img in enumerate(batch['src_image']):
                assert src_img.dtype == torch.uint8, f'Image dtype is not uint8, but {src_img.dtype}'
                img_pair = torch.cat([batch['src_image'][i], batch['trg_image'][i]], dim=2) # [c, h, w*2]
                img_pair = img_pair.permute(1, 2, 0).cpu().numpy().astype('uint8') # [h, w, c]
                Image.fromarray(img_pair).save(os.path.join('/tmp/where/to/download', f"{batch['src_media_id'][i]}-{batch['trg_media_id'][i]}.png"))


if __name__ == '__main__':
    main()
