# Streaming Dataset

A streaming dataset lib which loads data in a streaming fashion:
- Data samples (+ possibly their metadata) are stored as raw files
- The dataset downloads and yields them on the fly
- Supports constructing the dataset from a CSV/parquet/json index, file directory, CSV/parquet/json wildcard or "`split_file_texts.txt`"
- Decodes images/videos/audios through sample data processing callbacks (could be easily extended)
- Supports

# TODO
- [x] Index construction
- [ ] Dataset iterator
- [x] Shuffling each epoch
- [ ] Lazy index so that we can efficiently initialize large datasets on a single node without hitting disk space limits
- [x] Cache data + evict cold samples
- [ ] Video decoding
- [ ] Audio loading
- [ ] Tutorial/usage examples
- [ ] Resumal logic
- [ ] More test coverage
- [ ] Documentation
- [x] Support for data provides as callbacks (possibly via forward/backward translation)
- [ ] There is no global shuffling right now, so smth like ImageNet training will be flawed.
- [ ] Evict samples inside random access queries as well.
- [ ] Some addition/eviction race conditions might happen, when someone is evicting/downloading a sample which another worker is trying to get via random access.
- [ ] Fix TODOs in the codebase.
- [ ] Remove logging calls from the codebase.
- [ ] How to support multiple instances of the *same* dataset in a single process? That might lead to race conditions in downloading/eviction.

# Installation

```bash
pip install mosaicml-streaming beartype pytest torch torchvision torchaudio
```

# Usage
## Basic usage
```python
from sds.dataset import StreamingDataset
dataset = StreamingDataset(
    src='s3://snap-genvid-us-east-2/iskorokhodov/snapvideo_3_datasets/test_table/89c7c52fa90d4ee391ebbc39cd8ef5b9/000000000000.parquet',
    dst='ignore/tmp',
    data_type='image',
    columns_to_load=['data_url'],
    index_col_name='data_id',
    num_downloading_workers=10,
)
sample = dataset[0] # Downloads (with blocking) the sample and returns it
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=3, shuffle=False)
for batch in dataloader: # Loads samples in parallel.
    print(batch.keys())
```


## Running a simple demo
```bash
python scripts/unpack.py s3://snap-genvid-us-east-2/iskorokhodov/snapvideo_3_datasets/test_table/89c7c52fa90d4ee391ebbc39cd8ef5b9/000000000000.parquet ignore/tmp --columns_to_load data_url --index_col_name data_id --num_downloading_workers 10
```

# Running tests
```bash
PYTHONPATH=. pytest tests
```
