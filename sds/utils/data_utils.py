import os
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as fs
import pandas as pd
import polars as pl

#---------------------------------------------------------------------------

def read_parquet_slice(path: str, N: int, K: int, columns: list[str] = None, region: str = None) -> pd.DataFrame:
    """
    Load a row slice [N:N+K] from a Parquet file or dataset in a memory-efficient manner.
    """
    if path.startswith("s3://"):
        fsys = fs.S3FileSystem(region=region)
        path = path.replace("s3://", "")
    else:
        fsys = fs.LocalFileSystem()

    dataset = pq.ParquetDataset(path, filesystem=fsys)

    start_offset = N
    end_offset = N + K
    rows_accumulated = 0
    tables_to_concat = []

    for fragment in dataset.fragments:
        # THE FIX IS HERE: Use pq.ParquetFile() to get an object with row group metadata.
        pf = pq.ParquetFile(fragment.path, filesystem=fsys)

        for i in range(pf.num_row_groups):
            rg_meta = pf.metadata.row_group(i)
            rg_rows = rg_meta.num_rows

            if rows_accumulated + rg_rows < start_offset:
                rows_accumulated += rg_rows
                continue

            if rows_accumulated >= end_offset:
                break

            rg_table = pf.read_row_group(i, columns=columns)

            local_start = max(0, start_offset - rows_accumulated)
            local_end = min(rg_rows, end_offset - rows_accumulated)
            local_length = local_end - local_start

            if local_length > 0:
                tables_to_concat.append(rg_table.slice(local_start, local_length))

            rows_accumulated += rg_rows

        if rows_accumulated >= end_offset:
            break

    if not tables_to_concat:
        empty_schema = dataset.schema.to_arrow_schema()
        if columns:
            fields = [field for field in empty_schema if field.name in columns]
            empty_schema = pa.schema(fields)
        return pa.Table.from_pylist([], schema=empty_schema).to_pandas()

    result_table = pa.concat_tables(tables_to_concat)
    return result_table.to_pandas()

def save_polars_parquet(df: pd.DataFrame, dst: str, group_size: int = 100_000):
    """
    Saves a pandas DataFrame to a "Polars" Parquet file.
    TODO: I have no idea if polars saving is better than pandas saving in terms of pre-processing.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    df_pl = pl.from_pandas(df)
    df_pl.write_parquet(dst, use_pyarrow=True, row_group_size=group_size)

#---------------------------------------------------------------------------
