## Streaming Dataset for the People

A streaming dataset which fetches and yields samples on the fly, witch caching/eviction and random access. Features:
- *Very* flexible in terms of data sources:
  - Can be constructed from a CSV/parquet/json index, file directory, CSV/parquet/json wildcard or "`split_file_texts.txt`"
  - Supports remote and local data sources (e.g., S3, HTTP, local files, etc.)
- Has caching and eviction logic, so that you can efficiently work with large datasets without hitting disk space limits.
- Has standard data processing transforms for images, videos, audios, text, and metadata.
- Supports random access (through blocking calls)!
- Multi-stream dataloader (i.e., streaming from multiple data sources in parallel).

Based on: [snap-datastream](https://github.sc-corp.net/Snapchat/snap-datastream)

## Installation

Clone the repository and install the requirements:
```bash
pip install -r requirements.txt
```

## Usage
### Basic usage

ImageNet-1K data loading example:
```python
import torch
import torchvision.transforms.functional as TVF
from sds.dataset import StreamingDataset
from sds.transforms import presets

image_transforms = presets.create_standard_image_pipeline(
    image_field='jpeg',
    resolution=(256, 256),
)
metadata_transforms = presets.create_standard_metadata_pipeline(
    metadata_field='meta.json',
    class_label_metadata_field='class',
    one_hot_encode_to_size=1000,
    return_raw_metadata=True,
)

# The original init can take a bit of time, since it downloads the index metadata (400Mb in the case below).
dataset = StreamingDataset(
    # src='/nfs/datasets/imagenet-1k/train/',
    # src='s3://snap-genvid/datasets/imagenet-1k/train/',
    src='s3://snap-genvid/datasets/imagenet-1k/train.csv', # <= Equivalent to the above, but with precomputed index.
    # src='s3://snap-genvid/datasets/imagenet-1k/val.csv', # <= Equivalent to the above, but with precomputed index.
    dst='ignore/tmp',
    data_type='image',
    transforms=image_transforms + metadata_transforms,
    columns_to_download=['jpeg', 'meta.json'],
    index_col_name='index',
    num_downloading_workers=3,
    prefetch=100,
    shuffle_seed=42,
)
sample = dataset[0] # Synchronously downloads a sample and returns it
data_iterator = iter(torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=3))
for i in range(10):
    batch = next(data_iterator)
TVF.to_pil_image(batch['image'][0]) # Convert the first image to PIL for visualization
```

### Running a simple demo
```bash
python scripts/unpack.py s3://snap-genvid-us-east-2/iskorokhodov/snapvideo_3_datasets/test_table/89c7c52fa90d4ee391ebbc39cd8ef5b9/000000000000.parquet ignore/tmp --columns_to_download data_url --index_col_name data_id --num_downloading_workers 10
```

### Constructing an index
When training on files list directory (remote or local), it's recommended to precompute the index so that we don't need to scan the directory each time.
You can use a command like this to do that:
```bash
python scripts/construct_index.py --src s3://snap-genvid/datasets/tmp/ --dst s3://snap-genvid/datasets/tmp-index.csv --tmp_dir ignore/tmp --data_type image
```
After the index is constructed, you can simply replace your `src` argument with the index file path, e.g. `s3://snap-genvid/datasets/tmp-index.csv`.

## How it works
The entry point is the `StreamingDataset` class, which takes a source `src` and arguments and does the following:
1. It constructs an index from the source:
    - if `src` is a local or remote CSV/parquet/json file, it reads the index from there.
    - if `src` is a local or remote directory, it scans the directory and constructs an index from the files.
    - if `src` is a local or remote index wildcard (e.g., `/path/to/*.csv`), it scans the files matching the wildcard and constructs an index from them.
2. Then, we save the index on the node as a parquet file (for memory efficient chunked reading) and return an index metadata object.
3. After that, the dataset init initializes an "empty" downloader (without initializing the downloader workers). Without workers, the downloader can be used for random access queries, such as `dataset[0]`, which will download the sample with blocking.
3. When the iterator is created, we initialize the downloader workers, which will download samples in parallel and cache them on disk.
4. Then, each dataloading worker reads its index chunk, shuffles it and starts the generator, which yields samples one by one.
5. The downloader yields the indicies of the downloaded samples. Then, we look up the sample metadata by its index and process it through a sequence of sample processing callbacks (named transforms).
6. Sample transforms are very flexible and you can incorporate any processing logic you want, such as decoding images/videos/audios, applying augmentations, etc.
7. The are "presets" of sample transforms which should cover 80% of the cases for image/video/text-to-video/etc use cases.
8. Caching and eviction logic is performed by the StreamingDataset class, which keeps track of downloaded file sizes and evicts the oldest ones when the cache size exceeds the threshold. Currently, the cache size is set naively per workers as `node_cache_size / num_workers`, assuming that each worker has equal load.

## Contributing

### Current TODOs
- [x] Index construction
- [x] Dataset iterator
- [x] Shuffling each epoch
- [ ] Lazy index so that we can efficiently initialize large datasets on a single node without hitting disk space limits
- [x] Cache data + evict cold samples
- [x] Video decoding
- [x] Audio loading
- [ ] Tutorial/usage examples
- [x] Resumal logic. Only if the number of ranks is not changed, since otherwise, we will have shuffling discrepancies.
- [ ] More test coverage: state dict, resumal, index construction, deadlocks, sync-ed dataloader, etc.
- [ ] Documentation
- [x] Support for data provides as callbacks (possibly via forward/backward translation)
- [x] There is no global shuffling right now, so smth like ImageNet training will be flawed.
- [ ] Evict samples inside random access queries as well.
- [ ] Some addition/eviction race conditions might happen, when someone is evicting/downloading a sample which another worker is trying to get via random access.
- [ ] Fix TODOs in the codebase.
- [x] Remove logging calls from the codebase.
- [ ] How to support multiple instances of the *same* dataset in a single process? That might lead to race conditions in downloading/eviction.
- [ ] We likely also need some node-level file lock to keep disk usage information for caching, since each new iterator instance is thinking that it's starting from scratch.
- [ ] Shutdown for num_workers > 0 is quite slow. Not sure why.
- [x] Clean broken samples from disk.
- [x] Time-based garbage collection.
- [x] Get/load state dict and make sure we resume from it.
- [x] Can we construct a remote S3 index in parallel?
- [x] Construct an index for a local/remote directory.
- [x] Sometimes, we can have less raw index files that nodes.
- [x] Missing fields should be populated in the dataloader or index meta or where? (I guess, they should automatically be filled with `None` in the index).
- [x] Re-slice indicies based on sample counts and number of nodes.
- [ ] Cache index: save to "cache" on S3 and load from cache (if present). Basically, if we are given a folder or split_file_paths.txt or *.csv, then we could save the index normally (though we should be careful about single-node vs multi-node cases).
- [ ] VAE latents loading.
- [ ] Video + .wav files loading (now we only support video files with embedded audio).
- [x] An option for interleaved indexing.
- [ ] Refresh index cache when restarting the dataloader? I.e. at least change the new size...
- [ ] Re-opening __iter__ for multi-stream dataloader would break the synchronization of stream types.
- [ ] Recompute sample_in_epoch based on the number of workers. I.e. sample_in_local_epoch => sample_in_global_epoch.
- [ ] Lazy index does not work with sample_in_epoch.
- [ ] We shouldn't need to reset the downloader after each iter_slice finish...

### TODOs for V2.5
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
