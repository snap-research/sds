import os
import time
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as fs
import pandas as pd
import polars as pl
import duckdb
from tqdm import tqdm
from loguru import logger

#---------------------------------------------------------------------------

def read_parquet_slice(path: str, start_offset: int, end_offset: int, step: int=1, columns: list[str] = None, region: str = None) -> pd.DataFrame:
    """
    Load a row slice [start_offset:end_offset:step] from a Parquet file or dataset in a memory-efficient manner.
    The path for step=1 is highly optimized to prevent reading unused row groups and to minimize memory allocations.
    """
    assert step >= 1, f"Step must be at least 1, got {step}."
    start_time = time.time()

    if path.startswith("s3://"):
        fsys = fs.S3FileSystem(region=region or os.environ.get("AWS_REGION"))
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
    elapsed = time.time() - start_time
    logger.debug(f"Read slice from {path} [{start_offset}:{end_offset}:{step}] in {elapsed:.2f} seconds, shape: {result_table.shape}")
    return result_table.to_pandas()


def save_polars_parquet(df: pd.DataFrame, dst: str, group_size: int = 100_000):
    """
    Saves a pandas DataFrame to a "Polars" Parquet file.
    TODO: I have no idea if polars saving is better than pandas saving in terms of pre-processing.
    """
    assert isinstance(df, pd.DataFrame), f"Input must be a pandas DataFrame, got {type(df)}."
    assert group_size > 0, f"Group size must be greater than 0, got {group_size}."
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        df[col] = df[col].replace(pd.NA, None)  # Convert pandas NA to None for Polars compatibility
        df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (float, int)) else x)  # Convert to string or None
    df_pl = pl.from_pandas(df)
    df_pl.write_parquet(dst, use_pyarrow=True, row_group_size=group_size)

#---------------------------------------------------------------------------

def read_parquet_num_rows(file_path: str, filesystem: fs.S3FileSystem) -> int:
    """Synchronously reads a single Parquet file's metadata for its row count."""
    return pq.read_metadata(file_path, filesystem=filesystem).num_rows

def count_parquet_rows_in_s3(s3_path: str, verbose: bool = True) -> int:
    """
    Counts rows in S3 Parquet files using a thread pool for concurrency.

    Args:
        s3_path: The full S3 path (e.g., 's3://bucket/prefix/').
        verbose: If True, enables debug logging and a progress bar.

    Returns:
        The cumulative row count.
    """
    s3_path = s3_path.replace('*.parquet', '')  # Remove wildcard if present
    try:
        parsed = urlparse(s3_path)
        if parsed.scheme != 's3' or not parsed.netloc:
            raise ValueError("Path must be a valid S3 URI (e.g., 's3://bucket/prefix').")
        bucket, prefix = parsed.netloc, parsed.path.lstrip('/')
    except ValueError as e:
        logger.error(f"Invalid S3 path: {e}")
        return 0

    if verbose:
        logger.debug(f"Finding files in {s3_path}...")

    pa_fs = fs.S3FileSystem()
    selector = fs.FileSelector(f"{bucket}/{prefix}", recursive=True)
    file_infos = pa_fs.get_file_info(selector)
    parquet_files = [f.path for f in file_infos if f.path.endswith(".parquet")]

    assert len(parquet_files) > 0, f"No Parquet files found in {s3_path}."

    if verbose:
        logger.debug(f"Found {len(parquet_files)} Parquet files. Counting rows...")

    with ThreadPoolExecutor() as executor:
        # Pre-fill the `filesystem` argument for the worker function.
        worker_func = partial(read_parquet_num_rows, filesystem=pa_fs)

        # executor.map is a concise, parallel equivalent to a for-loop.
        results = executor.map(worker_func, parquet_files)

        if verbose:
            results = tqdm(results, total=len(parquet_files), desc="Processing files")

        return sum(results)

def maybe_run_sql_query_on_dataframe(df: pd.DataFrame, sql_query: str | None = None) -> pd.DataFrame:
    """
    Safely applies a SQL query to a pandas DataFrame using an isolated DuckDB connection.

    This function is designed to be "fork-safe" and avoid deadlocks in multiprocessing
    environments (e.g., Gunicorn, Celery) by creating a new, ephemeral DuckDB
    connection for each call.

    Args:
        df: The input pandas DataFrame.
        sql_query: The SQL query string to execute. The query must use the
                   keyword 'dataframe' to refer to the DataFrame.

    Returns:
        A new pandas DataFrame containing the results of the query, or the
        original DataFrame if sql_query is None.

    Raises:
        AssertionError: If inputs are invalid (not a DataFrame, query missing 'dataframe').
        duckdb.Error: Propagates any errors from the DuckDB query execution.
    """
    DATAFRAME_KEYWORD = 'dataframe'

    # 1. Handle the case of no query
    if sql_query is None:
        return df

    # 2. Perform original input validations
    assert isinstance(df, pd.DataFrame), f"SQL filtering is only supported for pandas DataFrames, got {type(df)}."
    assert DATAFRAME_KEYWORD in sql_query, f"SQL query must contain '{DATAFRAME_KEYWORD}' to indicate the DataFrame to operate on. Got: {sql_query}"

    logger.debug(f"Preparing to run SQL query. Shape before: {df.shape}")
    logger.debug(f"Query: \"{sql_query}\"")

    result_df = None
    try:
        # 3. Create an isolated, in-memory connection.
        # This is the key to being fork-safe. The connection exists only for this function call.
        with duckdb.connect(database=':memory:', read_only=False) as con:
            # 4. Execute the query. DuckDB's `query` method can automatically detect
            # and use local variables (like `df`) if they are named in the query.
            final_query = sql_query.replace(DATAFRAME_KEYWORD, 'df')

            # The query is executed, and the result is immediately converted to a pandas DataFrame.
            result_df = con.execute(final_query).df()
    except duckdb.Error as e:
        logger.error(f"A DuckDB error occurred while executing the query: {e}")
        raise # Re-raise the exception to allow the calling code to handle the failure.
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise # Also re-raise unexpected errors.

    logger.debug(f"Shape after SQL query: {result_df.shape}")

    # 5. Return the new, transformed DataFrame
    return result_df

#---------------------------------------------------------------------------
