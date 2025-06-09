# A (Truly) Streaming Dataset

```
We are building a beautiful DataLoader — absolutely incredible, folks. People come up to me, they say, “Sir, this is the most efficient, the most elegant DataLoader we’ve ever seen.” It’s true. We’re talking multi-worker, zero-copy, caching — it loads data so fast, your GPUs are going to say, “Please, slow down, I can’t keep up!” Others tried — they failed. Slow, clunky, full of bugs. Sad!

But not ours. Ours is simple. So simple, folks. You just plug it in — boom, it streams. Like magic. No configs, no ten-page docs, no graduate degree in YAML required. Just import and go. It’s that easy. Even a TensorFlow engineer could use it!

We’ve got shuffle — and I’m telling you, nobody’s ever seen randomness like this before. It’s so random, scientists at MIT called me. They said, “Sir, we ran a Kolmogorov-Smirnov test on your batches — it broke our servers.” That’s how random it is.

This isn’t your grandma’s shuffle. This is quantum chaos, folks. You think Fisher-Yates was good? Ours makes Fisher-Yates look like sorting alphabetically. It’s so fast, the indices don’t even know they’ve been shuffled — they find out after the forward pass. That’s how ahead of time we are.

People say to me, “Sir, is it even legal to shuffle this fast?” I don’t know. We’re checking with the lawyers. But it’s beautiful. It’s so fast and so random, even your loss function gets confused — and that’s good.

Even the collate function? It’s tremendous. Custom, modular, smart — not like those weak, lazy collate functions from Sleepy Torch 1.0. We’re prefetching data before you even knew you needed it. That’s how smart it is.

Some people call it magic, others call it torch genius. But I just call it: America’s DataLoader. Believe me.
```

A streaming dataset lib which loads data in a streaming fashion:
- Data samples (+ possibly their metadata) are stored as raw files
- The dataset downloads and yields them on the fly
- Supports constructing the dataset from a CSV table or just a file directory (possibly a remote one --- e.g., from S3)
- Works with images and videos
- Random access

# Running tests
```bash
PYTHONPATH=. pytest tests
```
