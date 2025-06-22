import os
from pyarrow import fs
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

#---------------------------------------------------------------------------

def read_parquet_slice(path: str, N: int, K: int, region: str = None) -> pa.Table:
    """
    Load a row slice [N:N+K] from a directory of Parquet files (local or S3).

    Args:
        path: Local path or 's3://...' S3 URI
        N: Start row index (inclusive)
        K: Number of rows to load
        region: AWS region if reading from S3

    Returns:
        pyarrow.Table with the requested row slice
    """
    is_s3 = path.startswith("s3://")

    if is_s3:
        # Strip scheme and set up pyarrow S3FileSystem
        stripped_path = path.replace("s3://", "")
        fsys = fs.S3FileSystem(region=region)
        dataset = pq.ParquetDataset(stripped_path, filesystem=fsys)
    else:
        fsys = fs.LocalFileSystem()
        dataset = pq.ParquetDataset(path, filesystem=fsys)

    start = N
    end = N + K
    rows_accum = 0
    batches = []

    for fragment in dataset.fragments:
        metadata = fragment.metadata
        num_rows = metadata.num_rows if metadata else fragment.physical_schema.num_rows

        if rows_accum + num_rows < start:
            rows_accum += num_rows
            continue
        if rows_accum >= end:
            break

        local_start = max(0, start - rows_accum)
        local_end = min(num_rows, end - rows_accum)

        table = fragment.to_table().slice(local_start, local_end - local_start)
        batches.append(table)

        rows_accum += num_rows
        if rows_accum >= end:
            break

    result_table = pa.concat_tables(batches) if batches else pa.table({})
    df = result_table.to_pandas()

    return df


def save_polars_parquet(df: pd.DataFrame, dst: str, group_size: int = 100_000):
    """
    Saves a pandas DataFrame to a "Polars" Parquet file.
    TODO: I have no idea if polars saving is better than pandas saving in terms of pre-processing.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    df_pl = pl.from_pandas(df)
    df_pl.write_parquet(dst, use_pyarrow=True, row_group_size=group_size)

#---------------------------------------------------------------------------
