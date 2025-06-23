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

# Running tests
```bash
PYTHONPATH=. pytest tests
```
