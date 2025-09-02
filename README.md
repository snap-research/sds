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

There is also a docker image for SnapVideo-V3 with streaming-dataset pre-installed: `440036398022.dkr.ecr.us-west-2.amazonaws.com/facecraft-ml:genai-video-aws-fa3-h100-torch271-126-sds`.

## Quickstart
Here is an example of how to use the streaming dataset for 2 streams.
Let's assume that the first stream is a remote S3 folder of videos, the second stream is given with index.parquet file.

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
recompute: true # Whether to recompute the index even if it already exists in S3.
val_ratio: 0.1 # The fraction of the dataset to use for validation dataset.
max_num_val_rows: 10000 # The maximum number of rows in the validation dataset.
```
Note: make sure that `s3_destination_path` is in the correct AWS region for your future training job.
Otherwise, there might be problems when fetching parquet chunks from S3.

Then, install the script env and run the BQ export script:
```bash
pip install genml-training-tools # Install this within the first hour of job start
pip install --upgrade google-cloud google-cloud-bigquery google-cloud-storage db-dtypes pandas pyarrow s3fs loguru pydantic PyYAML boto3 google-cloud-bigquery-storage pyarrow
python scripts/construct_index_from_bq_query.py composeme-v2.yaml
```
It will create a single parquet index file and upload it to S3.
It will also create a validation index file with `val_ratio` fraction of the rows (up to `max_num_val_rows`).

### Initializing the dataset

TBD :|

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
- [ ] Tutorial/usage examples
- [ ] Documentation
- [ ] Video + .wav files loading (now we only support video files with embedded audio).
- [ ] Tensor parallel support: iterating the streams from one dataloader for one meta-iter and broadcasting them within the group.
- [ ] For non-lazy parquet index without slicing and filtering, we don't need to reload-resave it.
- [ ] Fix the current unit tests.
- [ ] First select a caption embedding, then download the selected one for traffic optimization.
- [ ] We can download video chunks from S3 give the random offset/num frames we need.

### TODOs for v1.5:
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

### TODOs for v2
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
