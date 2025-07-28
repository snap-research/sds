import os
import argparse
import pandas as pd

from sds.utils import os_utils
from sds.index import maybe_shuffle_df

def slice_val_set(input_path: str, output_path: str, num_rows: int, tmp_dir: str = "/tmp"):
    os_utils.download_file(input_path, os.path.join(tmp_dir, "input.parquet"), skip_if_exists=False)

    df = pd.read_parquet(input_path, engine='pyarrow')
    df = maybe_shuffle_df(df, shuffle_seed=42)
    df_train = df[:-num_rows]
    df_val = df[-num_rows:]

    train_tmp_path = os.path.join(tmp_dir, "train.parquet")
    val_tmp_path = os.path.join(tmp_dir, "val.parquet")
    df_train.to_parquet(train_tmp_path, index=False)
    df_val.to_parquet(val_tmp_path, index=False)

    os_utils.upload_file(train_tmp_path, input_path)
    os_utils.upload_file(val_tmp_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice validation set from a Parquet file.")
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to the input Parquet file.")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Path to save the sliced Parquet file.")
    parser.add_argument("-n", "--num_rows", type=int, required=True, help="Number of rows to slice.")

    args = parser.parse_args()

    slice_val_set(
        input_path=args.input_path,
        output_path=args.output_path,
        num_rows=args.num_rows
    )
