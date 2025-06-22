"""
A simple script to test the global logic of the Streaming Dataset.
It launched the dataset and stores the samples in a local directory.
"""
import os
import argparse

import torch
from streaming.base import StreamingDataLoader

from sds.dataset import StreamingDataset
from sds.structs import DataSampleType


def test_unpack_sds(src: str, dst: str, data_type: str, **kwargs):
    dataset = StreamingDataset(
        src=src,
        dst=dst,
        data_type=DataSampleType.from_str(data_type),
        **kwargs,
    )
    dataloader = StreamingDataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    os.makedirs(dst, exist_ok=True)

    for i, sample in enumerate(dataloader):
        print(sample.keys())
        sample_id = sample['__key__'][0]
        sample_path = os.path.join(dst, f"sample_{sample_id}.pt")
        # torch.save(sample, sample_path)
        # print(f"Saved sample {sample_id} to {sample_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unpack the Streaming Dataset.")
    parser.add_argument("src", type=str, help="Source path of the dataset (CSV or JSON file).")
    parser.add_argument("dst", type=str, help="Destination directory to store the unpacked samples.")
    parser.add_argument("--data_type", type=str, default='image', help="Type of data in the dataset (e.g., 'image', 'video').")
    parser.add_argument("--columns_to_load", type=str, default=None, help="Comma-separated list of columns to use from the index file.")
    parser.add_argument("--index_col_name", type=str, default=None, help="Name of the index column in the dataset index file. Defaults to 'index'.")
    parser.add_argument("--num_downloading_workers", type=int, default=4, help="Number of workers to use for loading the dataset. Defaults to 0 (no parallel loading).")
    # parser.add_argument("--override", action='store_true', help="Override existing files in the destination directory.")
    args = parser.parse_args()

    test_unpack_sds(
        src=args.src,
        dst=args.dst,
        data_type=args.data_type,
        # override=args.override,
        columns_to_load=args.columns_to_load.split(',') if args.columns_to_load else None,
        index_col_name=args.index_col_name or 'index',
        num_downloading_workers=args.num_downloading_workers,
    )
