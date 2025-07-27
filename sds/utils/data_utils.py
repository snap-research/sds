import os
import pandas as pd
import polars as pl

#---------------------------------------------------------------------------

def read_parquet_slice(path: str, N: int, K: int) -> pd.DataFrame:
    """
    Efficiently load a row slice [N:N+K] from a directory of Parquet files using Polars.

    Args:
        path: Local path or 's3://...' S3 URI
        N: Start row index (inclusive)
        K: Number of rows to load

    Returns:
        polars.DataFrame with the requested row slice
    """
    # Polars can read from local or S3 paths directly
    # Ensure `s3fs` and `pyarrow` are installed for S3 access.

    lazy_df = pl.scan_parquet(path)

    sliced_df = (
        lazy_df
        .slice(offset=N, length=K)
        .collect(streaming=True)
    )

    return sliced_df.to_pandas()


def save_polars_parquet(df: pd.DataFrame, dst: str, group_size: int = 100_000):
    """
    Saves a pandas DataFrame to a "Polars" Parquet file.
    TODO: I have no idea if polars saving is better than pandas saving in terms of pre-processing.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    df_pl = pl.from_pandas(df)
    df_pl.write_parquet(dst, use_pyarrow=True, row_group_size=group_size)

#---------------------------------------------------------------------------
