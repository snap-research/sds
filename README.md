# A (Truly) Streaming Dataset

A streaming dataset lib which loads data in a streaming fashion:
- Data samples (+ possibly their metadata) are stored as raw files
- The dataset downloads and yields them on the fly
- Supports constructing the dataset from a CSV table or just a file directory (possibly a remote one --- e.g., from S3)
- Works with images and videos
