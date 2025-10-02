## Streaming Dataset for the People

A streaming dataset which fetches samples from anywhere and yields them on the fly, with caching/eviction and random access features:
- *Very* flexible in terms of data sources:
  - Can be constructed from a CSV/parquet/json index, file directory, CSV/parquet/json wildcard or good old "`split_file_texts.txt`"
  - Supports remote and local data sources (e.g., S3, GCS, HTTP, local files, etc.)
- Has caching and eviction logic, so that you can efficiently work with large datasets without hitting disk space limits.
- Has standard data processing transforms for images, videos, audios, text, and metadata.
- Supports random access (through blocking calls)!
- Has a built-int multi-stream dataloader (i.e., streaming from multiple data sources in parallel) with various mixing strategy between the streams.

## Installation

The package is available on the corporate PyPI:
```bash
pip install streaming-dataset
```

There is also a docker image for SnapVideo-V3 with streaming-dataset pre-installed:
- `streaming-dataset==0.4.3`: `440036398022.dkr.ecr.us-west-2.amazonaws.com/facecraft-ml:genai-video-aws-fa3-h100-torch271-126-sds-0-4-3`
- `streaming-dataset==0.4.6`: `440036398022.dkr.ecr.us-west-2.amazonaws.com/facecraft-ml:genai-video-aws-fa3-h100-torch271-126-sds-0-4-6`

## Quickstart
Here is an example of how to use the streaming dataset for 2 streams.
Let's assume that the first stream is a remote S3 folder of videos, the second stream is given with index.parquet file.

For the first stream, we would be loading it directly from S3, but for the second stream, we'll go through the process of preparing the data from BigQuery.

### Preparing the data from BigQuery
First, create a simple config for your output BQ table, i.e. `composeme-v2.yaml`:
```yaml
bq_project: "research-prototypes"
sql_query: "
  SELECT *
  FROM `research-prototypes.generative_ai_data_platform_test.personalization_getty_dataset`
  WHERE aes_score > 4
  AND caption IS NOT NULL
  AND getty_caption IS NOT NULL
  AND getty_title IS NOT NULL
  AND Body2DPoseHandsFace IS NOT NULL
  AND InstanceSegmentation IS NOT NULL
  AND FashionSegmentationImaterialist IS NOT NULL
  ORDER BY RAND()
  "
s3_destination_path: s3://snap-genvid/datasets/sds-index-files/composeme-v2.parquet
s3_bucket_region: us-west-2 # The region of the S3 bucket. Can be left empty, but would lead to an error in case of a mismatch between $AWS_REGION in the env and the actual region of the bucket.
recompute: true
val_ratio: 0.1 # The fraction of the dataset to use for validation dataset.
max_num_val_rows: 10000 # The maximum number of rows in the validation dataset.
local_tmp_dir: ~ # Local temporary directory where the merged parquet will be saved to if provided (needed for large 20M+ rows outputs). You can likely use `/lssd/index-exports-tmp`.
gcs_tmp_dir: ~ # Where to save intermediate results (needed for huge 70M+ rows outputs). You can likely use `gs://dlahiri/index-exports-tmp`
row_group_size: 20000 # Number of rows per parquet row group.
```
Note: make sure that `s3_destination_path` is in the correct AWS region for your future training job.
Otherwise, there might be problems when fetching parquet chunks from S3.

Also, it's important to specify `local_tmp_dir`/`gcs_tmp_dir` to push the intermediate results through for large files (20M+ rows or 100GB+).
Otherwise, the job might fail either becase S3 multi-part upload would attempt to use too many parts (over 10K limit) or BQ auth token would expire.

Then, install the script env and run the BQ export script:
```bash
pip install genml-training-tools # Install this within the first hour of job start
pip install --upgrade google-cloud google-cloud-bigquery google-cloud-storage db-dtypes pandas pyarrow s3fs loguru pydantic PyYAML boto3 google-cloud-bigquery-storage pyarrow
python scripts/construct_index_from_bq_query.py composeme-v2.yaml
```
It will create a single parquet index file and upload it to S3.
It will also create a validation index file with `val_ratio` fraction of the rows (up to `max_num_val_rows`).

### Minimal example with an S3 parquet index

There are multiple examples for different modalities in the `examples/` folder.
For brevity, we won't repeat them here, and just show some basic usage with a single stream.

```python
import os
from typing import Callable
os.environ['AWS_REGION'] = 'us-west-2' # Make sure that AWS_REGION is set to the region of your S3 bucket.
os.environ['SDS_LOG_LEVEL'] = 'DEBUG' # Set the log level to DEBUG to see more logs.

import torch
from PIL import Image
from sds.dataset import StreamingDataset
from sds.transforms import presets

def build_transforms() -> list[Callable]:
    # SDS contains some standard transforms for common use cases. Let's use the one for images.
    image_transforms = presets.create_standard_image_pipeline(
        image_field='personalization_image_path', # Where the downloading field is located.
        resolution=(256, 256),  # Resize the image to this resolution.
    )

    # Let's only keep the 'image' and 'media_id' fields in the final output, since there is a lot of unused stuff in the index file.
    # (This unused stuff can break pytorch's default collate function.)
    sample_cleaning_transform = presets.EnsureFieldsTransform(fields_whitelist=['image', 'media_id'], drop_others=True)

    return image_transforms + [HorizontalFlipTransform(), sample_cleaning_transform]


class HorizontalFlipTransform:
    # A simple horizontal flip transform just for illustration purposes
    # Transform is just a class with a __call__ method which takes a sample dict and returns a sample dict.
    def __call__(self, sample: dict) -> dict:
        assert 'image' in sample, f'Image field is missing in the sample: {sample.keys()}'
        assert isinstance(sample['image'], torch.Tensor), f'Image is not a torch.Tensor, but {type(sample["image"])}'
        assert sample['image'].ndim == 3, f'Wrong image shape: {sample["image"].shape}'
        assert sample['image'].shape[0] in [1, 3], f'Wrong image shape: {sample["image"].shape}'
        sample['image'] = torch.flip(sample['image'], dims=[2]) if torch.rand(1) < 0.5 else sample['image'] # [c, h, w]
        return sample


def main():
    dataset = StreamingDataset(
        src='s3://snap-genvid/datasets/sds-index-files/composeme-v2.parquet',
        dst='/tmp/where/to/download',      # Where to download the samples.
        transforms=build_transforms(),    # A list of transforms to apply.

        # Which columns to download from the index. Their values should be URLs/paths.
        # Once downloaded, a sample dict will be constructed, with the column names pointing to the local paths.
        # After that, all the transforms will be applied to the sample dict.
        columns_to_download=['personalization_image_path', 'personalization_caption_path'],

        index_col_name='media_id',         # Name of the column to use as the index (should have unique values for samples).
        num_downloading_workers=5,         # How many parallel downloading workers to use.
        prefetch=100,                      # How many samples to pre-download ahead of time.
        cache_limit='10gb',                # How much disk space to use for caching.
        allow_missing_columns=False,       # Should we ignore or not ignore samples with some missing `columns_to_download`?

        # Some configuration for the `lazy index`. Sometimes, the index file is huge, so it's better
        # not to load it all at once, but rather fetch in chunks.
        lazy_index_chunk_size=1000,         # Chunk size to fetch.
        lazy_index_num_threads=2,           # In how many threads to fetch the index chunks.
        lazy_index_prefetch_factor=3,       # How many chunks to prefetch ahead of time.
        min_num_pending_tasks_thresh=200,   # How many downloading samples should there be pending, before we start scheduling for sample downloading the next index chunk.

        # Shuffle seed to use. If None, no shuffling is performed.
        # If -1, then a random seed will be created on the first rank and distributed across all the ranks.
        shuffle_seed=123,

        # Let's filter out the images without any related images.
        # The SQL query is quite arbitrary and applied on each index chunk before scheduling downloading tasks.
        sql_query="SELECT * FROM dataframe WHERE num_related_images > 0",

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
    Image.fromarray(sample['image'].permute(1, 2, 0).cpu().numpy().astype('uint8')).save('/tmp/where/to/download/debug-sample-12345.png')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=2, pin_memory=True, drop_last=True)
    data_iterator = iter(dataloader)

    for epoch in range(1, 2):
        for i in range(7):
            batch = next(data_iterator)
            print(f'Processing batch #{i + 1} with {len(batch["media_id"])} items.')
            for i, img in enumerate(batch['image']):
                assert img.dtype == torch.uint8, f'Image dtype is not uint8, but {img.dtype}'
                img = img.permute(1, 2, 0).cpu().numpy().astype('uint8') # [h, w, c]
                Image.fromarray(img).save(os.path.join('/tmp/where/to/download', batch['media_id'][i] + '.png'))


if __name__ == '__main__':
    main()
```

### (Very) minimal example with folder datasets
For this example, we are just iterating over the text files in a local folder.
The nasty part of the folder datasets is the necessity to specify a `data_type`.

```python
import os
import torch
from sds.dataset import StreamingDataset

src = '/tmp/dummy-data' # Could, in fact, be an S3 path as well (though harder to generate the data for)
dst = '/tmp/dummy-out'
EXT = 'txt'

# Generate some dummy data.
os.makedirs(src, exist_ok=True) # Let's generate some dummy data first.
for i in range(10):
    with open(os.path.join(src, f'{i:05d}.{EXT}'), 'w') as f:
        f.write(f'This is sample {i}.\n' * 10)

class LoadTransform():
    def __call__(self, sample: dict) -> dict:
        # We got a sample with keys: `index` (filename) and `txt` (path to the text file, inferred from the extension).
        with open(sample[EXT], 'r') as f:
            sample[EXT] = f.read() # Load the text content.
        return sample

# For folder datasets, `columns_to_download` specify the sample data to copy (from local or S3).
dataset = StreamingDataset(src, dst, columns_to_download=[EXT], data_type='text', transforms=[LoadTransform()])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)
for i, batch in enumerate(dataloader):
    print(f'Batch {i}', batch)
```


### Debugging tips

- Set `num_workers=0` in the dataloader.
- Set `print_exceptions=True` and `print_traceback=True` in the dataset to see what is going wrong.

## How it works
The entry point is the `StreamingDataset` class, which takes a source `src` and arguments and does the following:
1. It constructs an index from the source:
    - if `src` is a local or remote CSV/parquet/json file, it reads the index from there.
    - if `src` is a local or remote directory, it scans the directory and constructs an index from the files.
    - if `src` is a local or remote index wildcard (e.g., `/path/to/*.csv`), it scans the files matching the wildcard and constructs an index from them.
2. Then, if it's a lazy index (controlled via `lazy_index_chunk_size`), we broadcast the index metadata to each rank. If not lazy, we save the index on the node as a parquet file (for memory efficient chunked reading) and return an index metadata object.
3. After that, the dataset init initializes an "empty" downloader (without initializing the downloader workers). Without workers, the downloader can be used for random access queries, such as `dataset[0]`, which will download the sample with blocking.
3. When the iterator is created, we initialize the downloader workers, which will download samples in parallel and cache them on disk.
4. Then, each dataloading worker reads its index chunk, shuffles it and starts the generator, which yields samples one by one. If it's a lazy index, then each worker would be reading index chunks with some prefetching (via downloading threads).
5. The downloader yields the indicies of the downloaded samples. Then, we look up the sample metadata by its index and process it through a sequence of sample processing callbacks (named transforms).
6. Sample transforms are very flexible and you can incorporate any processing logic you want, such as decoding images/videos/audios, applying augmentations, etc.
7. The are "presets" of sample transforms which should cover 80% of the cases for image/video/text-to-video/etc use cases.
8. Caching and eviction logic is performed by the StreamingDataset class, which keeps track of downloaded file sizes and evicts the oldest ones when the cache size exceeds the threshold. Currently, the cache size is set naively per workers as `node_cache_size / num_workers`, assuming that each worker has equal load.

## Contributing

### TODOs for v1
- [x] Index construction
- [x] Dataset iterator
- [x] Shuffling each epoch
- [x] Lazy index so that we can efficiently initialize large datasets on a single node without hitting disk space limits
- [x] Cache data + evict cold samples
- [x] Video decoding
- [x] Audio loading
- [x] Resumal logic. Only if the number of ranks is not changed, since otherwise, we will have shuffling discrepancies.
- [x] There is no global shuffling right now, so smth like ImageNet training will be flawed.
- [x] Remove logging calls from the codebase.
- [x] Clean broken samples from disk.
- [x] Time-based garbage collection.
- [x] Get/load state dict and make sure we resume from it.
- [x] Can we construct a remote S3 index in parallel?
- [x] Construct an index for a local/remote directory.
- [x] Sometimes, we can have less raw index files that nodes.
- [x] Missing fields should be populated in the dataloader or index meta or where? (I guess, they should automatically be filled with `None` in the index).
- [x] Re-slice indicies based on sample counts and number of nodes.
- [x] VAE latents loading.
- [x] An option for interleaved indexing.
- [x] Re-opening __iter__ for multi-stream dataloader would break the synchronization of stream types.
- [x] Lazy index does not work with sample_in_epoch.
- [x] We shouldn't need to reset the downloader after each iter_slice finish...
- [x] For lazy index, schedule next index chunk before the current one is finished.
- [x] Make MultiStreamDataLoader robust to re-opening the iterator.
- [x] Docker image.
- [x] Mixing between streams across ranks.
- [x] BQ script with exportion into a single parquet file.
- [x] Video latents loading.
- [x] Fixed random order.
- [x] Consecutive interleaved order.
- [x] Put on our corp pypi.
- [x] Evict samples inside random access queries as well.
- [x] Our caching logic is broken: we think we've downloaded a sample and occupied some disk space, but it was already there. This makes us delete samples thinking that we need to free up space.
- [x] Support shuffle_seed = -1.
- [x] Audio normalization.
- [x] Row group size = 20k for the new script.
- [x] Tutorial/usage examples
- [ ] Documentation
- [ ] Video + .wav files loading (now we only support video files with embedded audio).
- [ ] Tensor parallel support: iterating the streams from one dataloader for one meta-iter and broadcasting them within the group.
- [ ] For non-lazy parquet index without slicing and filtering, we don't need to reload-resave it.
- [ ] Fix the current unit tests.
- [ ] `pre_download_transforms` to first select a caption embedding, then downloading the selected one for traffic optimization.
- [ ] How can we reweight the index during training? A straightforward way would be randomly filtering out samples in the index via SQL queries. But maybe, we can have a reweighting_fn as an input or a weight column in the index?
- [ ] Support spawn start method for dataloader workers.
- [ ] An option to cache the downloaded/loaded sample dict? Ideally, through some cache transform, i.e. so we can cache at any selected point in the transform chain. Then, we can store videos unpacked as np4/torch files and load them much faster.
- [ ] `sds.utils.data_utils.read_parquet_slice` is not working for a wildcard of parquets.

### TODOs for v1.5:
- [ ] Is it possible to make `construct_index_from_bq_small.py` work for large tables? It's logic is much cleaner...
- [ ] Fix random seeds in transforms. Possibly, by adding a `__random_seed__` field? Or would fixing a global random seed be enough?
- [ ] The logic for resetting the downloader after each epoch is hacky. I dont think we should do that.
- [ ] More test coverage: state dict, resumal, index construction, deadlocks, sync-ed dataloader, etc.
- [ ] Shutdown for num_workers > 0 is quite slow. Not sure why.
- [ ] Recompute sample_in_epoch based on the number of workers. I.e. sample_in_local_epoch => sample_in_global_epoch.
- [ ] @beartype for streaming dataset init method.
- [ ] Allow empty columns_to_download (i.e., only metadata).
- [ ] Refresh index cache when restarting the dataloader? I.e. at least check the new size...
- [ ] Support for data providers as callbacks (possibly via forward/backward translation)
- [ ] Cache index: save to "cache" on S3 and load from cache (if present). Basically, if we are given a folder or split_file_paths.txt or *.csv, then we could save the index normally (though we should be careful about single-node vs multi-node cases).
- [ ] Deterministic order for the thread pool downloader.
- [ ] Some race conditions might happen, when someone is evicting/downloading a sample with a downloader, while someone else is doing this via random access, since random access breaks the non-overlapping assumption. Also, we don't free the disk space used by random access samples. We should probably lock the downloader (among all the node workers?!) during random access queries.
- [ ] How to support multiple instances of the *same* dataset in a single process? That might lead to race conditions in downloading/eviction.
- [ ] We likely also need some node-level file lock to keep disk usage information for caching, since each new iterator instance is thinking that it's starting from scratch.
- [ ] Plot the job cost in terms of downloading, given the number of requests, and bytes downloaded.

### TODOs for v2
- [ ] We can download video chunks from S3 given the random offset/num frames we need.
- [ ] Fix TODOs in the codebase (i.e. grep for "TODO" and fix).
- [ ] SQLite index instead of parquet.
- [ ] Move synchronous batch-wise yielding to the `StreamingDataset` class using the round-robin assumption of torch dataloader iterating over workers.

### Running tests
```bash
PYTHONPATH=. pytest tests
```

### Style guide
Create your own branch, make changes, and create a pull request.

Style guide:
- Use rebase instead of merges where possible.
- 4 spaces for indentation
- Always annotate shapes of tensors via inline comments (even in the obvious cases), e.g.:
```python
x = torch.randn(3, 4, 5) # [batch_size, sequence_length, hidden_size]
```
- Always annotate the type of the function arguments and return values.
- We use something similar to [AngularJS commit styleguide](https://gist.github.com/brianclements/841ea7bffdb01346392c): a commit should be of the form `<type>(<scope>): <subject>`, where `<type>` is one of the following:
  - `feat`: a new feature
  - `fix`: a bug fix
  - `docs`: changes to documentation
  - `style`: formatting, missing semi colons, etc; no code change
  - `refactor`: refactoring production code
  - `test`: adding tests, refactoring test; no production code change
  - `chore`: updating build tasks, package manager configs, etc; no production code change
  - `revert`: reverting to a previous commit
  - `perf`: a code change that improves performance
  - `ci`: changes to CI configuration files and scripts
  - `build`: changes that affect the build system or external dependencies
  - `temp`: temporary commit that won't be included in the final version
  - `wip`: work in progress
