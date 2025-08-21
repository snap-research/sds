"""
Constructs an index for local/remote files directory.
Is convenient so not to scan later the whole directory at train/test time.
"""

import os
import argparse

from sds.index import build_index, load_index_partition
from sds.structs import DataSampleType
from sds.utils import os_utils

#---------------------------------------------------------------------------

def construct_index(src: str, dst: str, tmp_dir: str, data_type: str):
    index_meta = build_index(src=src, dst_dir=tmp_dir, data_type=DataSampleType.from_str(data_type), shuffle_seed=None)
    index = load_index_partition(index_meta, rank=0, num_ranks=1, num_nodes=1)
    tmp_index_path = os.path.join(tmp_dir, 'index.csv')
    index.to_csv(tmp_index_path, index=False)
    os_utils.upload_file(tmp_index_path, dst)

#---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct an index for local/remote files directory.")
    parser.add_argument('--src', type=str, required=True, help='Source path or URL to the dataset index.')
    parser.add_argument('--dst', type=str, required=True, help='Destination path to save the index.')
    parser.add_argument('--tmp_dir', type=str, required=True, help='Temporary directory to store intermediate files.')
    parser.add_argument('--data_type', type=str, required=True, help='Type of the main data samples in the dataset.')

    args = parser.parse_args()

    construct_index(
        src=args.src,
        dst=args.dst,
        tmp_dir=args.tmp_dir,
        data_type=args.data_type,
    )

#---------------------------------------------------------------------------
