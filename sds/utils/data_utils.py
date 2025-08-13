import os
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as fs
import pandas as pd
import polars as pl
from loguru import logger

#---------------------------------------------------------------------------

def read_parquet_slice(path: str, start_offset: int, end_offset: int, step: int=1, columns: list[str] = None, region: str = None) -> pd.DataFrame:
    """
    Load a row slice [start_offset:end_offset:step] from a Parquet file or dataset in a memory-efficient manner.
    The path for step=1 is highly optimized to prevent reading unused row groups and to minimize memory allocations.
    """
    assert step >= 1, f"Step must be at least 1, got {step}."

    if path.startswith("s3://"):
        fsys = fs.S3FileSystem(region=region)
        path = path.replace("s3://", "")
    else:
        fsys = fs.LocalFileSystem()

    dataset = pq.ParquetDataset(path, filesystem=fsys)
    rows_accumulated = 0
    tables_to_concat = []

    for fragment in dataset.fragments:
        pf = pq.ParquetFile(fragment.path, filesystem=fsys)

        for i in range(pf.num_row_groups):
            rg_meta = pf.metadata.row_group(i)
            rg_rows = rg_meta.num_rows

            # This logic efficiently skips row groups entirely outside the slice boundaries.
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
                if step == 1:
                    # For step=1, use the original, highly memory-efficient logic.
                    # This avoids creating intermediate index arrays and uses a simple slice.
                    tables_to_concat.append(rg_table.slice(local_start, local_length))
                else:
                    # For step > 1, use the separate logic which is slightly less memory-efficient
                    # but correctly handles stepping.
                    contiguous_rg_slice = rg_table.slice(local_start, local_length)

                    global_start_of_slice = rows_accumulated + local_start
                    offset_from_global_start = global_start_of_slice - start_offset
                    first_local_index = (step - (offset_from_global_start % step)) % step

                    indices_to_take = pa.array(range(first_local_index, len(contiguous_rg_slice), step))

                    if len(indices_to_take) > 0:
                        stepped_table = contiguous_rg_slice.take(indices_to_take)
                        tables_to_concat.append(stepped_table)

            rows_accumulated += rg_rows

        if rows_accumulated >= end_offset:
            break

    if not tables_to_concat:
        empty_schema = dataset.schema
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
    assert isinstance(df, pd.DataFrame), f"Input must be a pandas DataFrame, got {type(df)}."
    assert group_size > 0, f"Group size must be greater than 0, got {group_size}."
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(value=None) # Filling np.nan with None for non-numeric columns.
    df_pl = pl.from_pandas(df).with_columns([pl.col(c).cast(pl.String) for c in object_cols])
    df_pl.write_parquet(dst, use_pyarrow=True, row_group_size=group_size)

#---------------------------------------------------------------------------
