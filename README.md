# Streaming Dataset

A streaming dataset lib which loads data in a streaming fashion:
- Data samples (+ possibly their metadata) are stored as raw files
- The dataset downloads and yields them on the fly
- Supports constructing the dataset from a CSV table or just a file directory (possibly a remote one --- e.g., from S3)
- Works with images and videos
- Random access

# TODO
- [x] Index construction
- [ ] Dataset iterator
- [x] Shuffling each epoch
- [ ] Lazy index so that we can efficiently initialize large datasets on a single node without hitting disk space limits
- [ ] Cache data + evict cold samples
- [ ] Video decoding
- [ ] Audio loading
- [ ] Tutorial/usage examples
- [ ] Resumal logic
- [ ] More test coverage
- [ ] Documentation
- [ ] Support for data provides as callbacks (possibly via forward/backward translation)
- [ ] For random access, we might have a problem, when two workers try to access the same file at the same time. We need to implement some kind of locking mechanism?
- [ ] There is no global shuffling right now, so smth like ImageNet training will be flawed.

# Running tests
```bash
PYTHONPATH=. pytest tests
```
