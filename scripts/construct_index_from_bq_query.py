# """
# env/usage:
# # pip install genml-training-tools -> Install this within the first hour of job start
# # pip install --upgrade google-cloud google-cloud-bigquery google-cloud-storage db-dtypes pandas pyarrow s3fs loguru pydantic PyYAML boto3 google-cloud-bigquery-storage pyarrow
# # clear && python construct_index_from_bq_query.py --config construct_index_from_bq_query.yaml
# """

# import os
# import argparse
# from urllib.parse import urlparse
# import io
# from typing import Iterator

# import yaml
# import s3fs
# from google.cloud import bigquery
# import pyarrow.parquet as pq
# import pyarrow as pa
# import pandas as pd
# from loguru import logger
# from google.cloud import bigquery_storage
# from tqdm import tqdm

# from google.cloud import bigquery
# from genml_training_tools.gcp.credentials import get_default_credentials

# """
# env/usage:
# # pip install genml-training-tools -> Install this within the first hour of job start
# # pip install --upgrade google-cloud google-cloud-bigquery google-cloud-storage db-dtypes pandas pyarrow s3fs loguru pydantic PyYAML boto3 google-cloud-bigquery-storage
# # clear && python construct_index_from_bq_query.py --config construct_index_from_bq_query.yaml
# """

import math
import argparse
from typing import Iterator

import yaml
import s3fs
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import bigquery, bigquery_storage
from loguru import logger
from tqdm import tqdm

from genml_training_tools.gcp.credentials import get_default_credentials

#---------------------------------------------------------------------------

def construct_index_from_bq_query(
    bq_project: str,
    sql_query: str,
    s3_destination_path: str,
    recompute: bool = False,
    num_val_rows: int = 10000
):
    assert s3_destination_path.startswith('s3://') and s3_destination_path.endswith(".parquet"), f"Invalid S3 path: {s3_destination_path}"
    s3 = s3fs.S3FileSystem()
    s3_fs_pa = pa.fs.S3FileSystem()
    val_s3_destination_path = s3_destination_path.replace(".parquet", "-val.parquet")

    if s3.exists(s3_destination_path) and not recompute:
        logger.info(f"File already exists at {s3_destination_path}. Skipping computation.")
        return

    # Set up BigQuery clients
    creds, _ = get_default_credentials()
    bq_client = bigquery.Client(credentials=creds, project=bq_project)
    bqstorage_client = bigquery_storage.BigQueryReadClient(credentials=creds)

    logger.info("Starting BigQuery query execution...")
    query_job = bq_client.query(sql_query)
    results = query_job.result() # This call blocks until the query completes and metadata is available.

    # Now we have the total row count for a meaningful progress bar!
    total_rows = results.total_rows
    logger.info(f"Query finished. Found {total_rows} rows to process.")

    if total_rows == 0:
        logger.warning("Query returned 0 rows. No Parquet file will be created.")
        return

    # Get the iterator for streaming results
    dataframe_iterator = results.to_dataframe_iterable(bqstorage_client=bqstorage_client)

    # Stream data directly to the S3 path
    build_parquet_from_chunks(
        dataframe_chunks_iterator=dataframe_iterator,
        destination_path=s3_destination_path.replace("s3://", ""),
        val_destination_path=val_s3_destination_path.replace("s3://", ""),
        filesystem=s3_fs_pa,
        total_rows=total_rows,
        num_val_rows=num_val_rows,
    )

    logger.info(f"Successfully wrote {total_rows} rows to {s3_destination_path}")


def build_parquet_from_chunks(
    dataframe_chunks_iterator: Iterator[pd.DataFrame],
    destination_path: str,
    val_destination_path: str,
    filesystem: pa.fs.S3FileSystem,
    total_rows: int,
    num_val_rows: int = 10000,
) -> None:
    """
    Iterates over dataframe chunks and writes them to a single Parquet file in a given filesystem.
    """
    val_rows_ratio = num_val_rows / total_rows
    num_val_rows_written = 0
    assert val_rows_ratio <= 0.5, f"num_val_rows ({num_val_rows}) must be less than half of total rows ({total_rows})."

    writer = None
    val_writer = None
    pbar = tqdm(total=total_rows, unit="rows", desc="Writing to S3")

    for chunk_df in dataframe_chunks_iterator:
        if chunk_df.empty:
            logger.warning("Skipping an empty dataframe chunk.")
            continue

        table_chunk = pa.Table.from_pandas(chunk_df)
        if writer is None:
            writer_kwargs = dict(schema=table_chunk.schema, compression='snappy', filesystem=filesystem)
            writer = pq.ParquetWriter(destination_path, **writer_kwargs)
            val_writer = pq.ParquetWriter(val_destination_path, **writer_kwargs) if num_val_rows > 0 else None

        # Determine the amount of validation data to write from this chunk
        num_val_chunk_rows = math.ceil(len(chunk_df) * val_rows_ratio)
        if val_writer is not None and num_val_chunk_rows > 0:
            val_table_chunk = table_chunk.slice(0, num_val_chunk_rows)
            val_writer.write_table(table=val_table_chunk)
            num_val_rows_written += num_val_chunk_rows
            table_chunk = table_chunk.slice(num_val_chunk_rows)

        if num_val_rows_written >= num_val_rows and val_writer is not None:
            logger.info(f"Wrote {num_val_rows_written} validation rows to {val_destination_path}. Closing validation writer.")
            val_writer.close()
            val_writer = None

        writer.write_table(table=table_chunk)
        pbar.update(len(chunk_df))

    if writer is None:
        logger.warning("The dataframe iterator was empty. No file was created.")

    if writer:
        writer.close()

#---------------------------------------------------------------------------
# Config utils.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BigQuery tables and upload to S3.")
    parser.add_argument('config', type=str, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    construct_index_from_bq_query(
        bq_project=config['bq_project'],
        sql_query=config['sql_query'],
        s3_destination_path=config['s3_destination_path'],
        recompute=config.get('recompute', False),
    )

#---------------------------------------------------------------------------
