"""
Unfortunately, this nice small script only works for small outputs (~<10M rows)...
For large files, use construct_index_from_bq.py, which is much more cumbersome :(
"""
import os, math, time, argparse, hashlib
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional, Tuple

import yaml
import s3fs
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import bigquery, bigquery_storage
from loguru import logger

from genml_training_tools.gcp.credentials import get_default_credentials


#----------------------------------------------------------
# Small helpers

def get_bq_clients(project: str, location: str) -> Tuple[bigquery.Client, bigquery_storage.BigQueryReadClient]:
    creds, _ = get_default_credentials()
    return (
        bigquery.Client(credentials=creds, project=project, location=location),
        bigquery_storage.BigQueryReadClient(credentials=creds),
    )


def get_s3_filesystems(region: Optional[str]) -> Tuple[s3fs.S3FileSystem, pa.fs.S3FileSystem]:
    return s3fs.S3FileSystem(), pa.fs.S3FileSystem(region=region)


def unique_tmp_table_id(project: str, dataset: str, sql_query: str) -> str:
    salt = f"{time.time_ns()}:{os.getpid()}:{uuid4().hex}"
    h = hashlib.sha256((sql_query + salt).encode()).hexdigest()[:16]
    return f"{project}.{dataset}.tmp_{h}"


def ensure_temp_table_with_ttl(bq_client: bigquery.Client, table_id: str, ttl_hours: int = 6) -> None:
    # Pre-create with expiration; ok if it already exists—job uses WRITE_TRUNCATE anyway.
    table = bigquery.Table(table_id)
    table.expires = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
    try:
        bq_client.create_table(table)
    except Exception:
        pass  # benign: dataset might auto-expire or table already exists


def run_query_to_table(bq_client: bigquery.Client, sql: str, table_id: str) -> None:
    cfg = bigquery.QueryJobConfig(destination=table_id,
                                  write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
    logger.info("Starting BigQuery query execution...")
    bq_client.query(sql, job_config=cfg).result()
    logger.info("BigQuery job completed.")


def table_row_count(bq_client: bigquery.Client, table_id: str) -> int:
    return int(bq_client.get_table(table_id).num_rows or 0)


def dataframe_iter_from_table(
    bq_client: bigquery.Client,
    bqstorage_client: bigquery_storage.BigQueryReadClient,
    table_id: str,
) -> Iterator[pd.DataFrame]:
    table = bq_client.get_table(table_id)
    return bq_client.list_rows(table).to_dataframe_iterable(bqstorage_client=bqstorage_client)


def build_parquet_from_chunks(
    dataframe_chunks_iterator: Iterator[pd.DataFrame],
    destination_path: str,
    filesystem: pa.fs.S3FileSystem,
    total_rows: int,
    val_ratio: float = 0.1,
    max_num_val_rows: int = 10_000,
    row_group_size: int = 20_000,
) -> None:
    if val_ratio == 0:
        num_val_rows = 0
        val_destination_path = None
        logger.warning("No validation set requested. This incident will be reported to Bobby.")
    else:
        val_destination_path = destination_path.replace(".parquet", "-val.parquet")
        num_val_rows = min(int(total_rows * val_ratio), max_num_val_rows)
        if num_val_rows > 100_000:
            logger.warning(f"Writing so many ({num_val_rows}) validation rows is too much. This incident will be reported to Evan.")
        logger.info(f"Will write {num_val_rows} validation rows to {val_destination_path}.")

    val_ratio = (num_val_rows / total_rows) if total_rows else 0.0
    assert val_ratio <= 0.5, f"num_val_rows ({num_val_rows}) must be < half of total rows ({total_rows})."

    writer = val_writer = None
    from tqdm import tqdm
    pbar = tqdm(total=total_rows, unit="rows", desc="Writing to S3")

    for chunk_df in dataframe_chunks_iterator:
        if chunk_df.empty:
            continue
        table_chunk = pa.Table.from_pandas(chunk_df)
        if writer is None:
            kw = dict(schema=table_chunk.schema, compression='snappy', filesystem=filesystem)
            writer = pq.ParquetWriter(destination_path, **kw)
            val_writer = pq.ParquetWriter(val_destination_path, **kw) if num_val_rows > 0 else None

        n_val = min(num_val_rows, math.ceil(len(chunk_df) * val_ratio))
        if val_writer is not None and n_val > 0:
            val_writer.write_table(table_chunk.slice(0, n_val))
            num_val_rows -= n_val
            table_chunk = table_chunk.slice(n_val)
            if num_val_rows <= 0:
                val_writer.close()
                val_writer = None

        writer.write_table(table_chunk, row_group_size=row_group_size)
        pbar.update(len(chunk_df))

    if writer:
        writer.close()


#----------------------------------------------------------
# Orchestration

def construct_index_from_bq_query(
    bq_project: str,
    sql_query: str,
    s3_destination_path: str,
    recompute: bool = False,
    s3_bucket_region: Optional[str] = None,
    bq_dataset_for_temp: str = "scratch",
    bq_location: str = "US",
    **writing_kwargs,
):
    assert s3_destination_path.startswith('s3://') and s3_destination_path.endswith(".parquet"), \
        f"Invalid S3 path: {s3_destination_path}"

    s3, s3_fs_pa = get_s3_filesystems(s3_bucket_region)

    if s3.exists(s3_destination_path) and not recompute:
        logger.info(f"File already exists at {s3_destination_path}. Skipping computation.")
        return

    bq_client, bqstorage_client = get_bq_clients(bq_project, bq_location)
    tmp_table = unique_tmp_table_id(bq_project, bq_dataset_for_temp, sql_query)

    # Create temp table with TTL; always cleanup in finally.
    ensure_temp_table_with_ttl(bq_client, tmp_table, ttl_hours=6)
    try:
        run_query_to_table(bq_client, sql_query, tmp_table)

        total_rows = table_row_count(bq_client, tmp_table)
        logger.info(f"Query finished. Found {total_rows} rows to process.")
        if total_rows == 0:
            logger.warning("Query returned 0 rows. No Parquet file will be created.")
            return

        df_iter = dataframe_iter_from_table(bq_client, bqstorage_client, tmp_table)
        build_parquet_from_chunks(
            dataframe_chunks_iterator=df_iter,
            destination_path=s3_destination_path.replace("s3://", ""),
            filesystem=s3_fs_pa,
            total_rows=total_rows,
            **writing_kwargs,
        )

        logger.info(f"Successfully wrote {total_rows} rows to {s3_destination_path}")
    finally:
        try:
            bq_client.delete_table(tmp_table, not_found_ok=True)
            logger.info(f"Deleted temp table {tmp_table}.")
        except Exception as e:
            logger.warning(f"Failed to delete temp table {tmp_table}: {e}")


#----------------------------------------------------------
# CLI

def main():
    parser = argparse.ArgumentParser(description="Prepare BigQuery tables and upload to S3.")
    parser.add_argument('config', type=str, help='Path to the YAML configuration file.')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    construct_index_from_bq_query(
        bq_project=cfg['bq_project'],
        sql_query=cfg['sql_query'],
        s3_destination_path=cfg['s3_destination_path'],
        recompute=cfg.get('recompute', False),
        max_num_val_rows=cfg.get('max_num_val_rows', 10000),
        val_ratio=cfg.get('val_ratio', 0.1),
        s3_bucket_region=cfg.get('s3_bucket_region'),
        bq_dataset_for_temp=cfg.get('bq_dataset_for_temp', 'scratch'),
        bq_location=cfg.get('bq_location', 'US'),
    )

if __name__ == "__main__":
    main()
