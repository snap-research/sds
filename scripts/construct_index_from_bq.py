import os, math, time, argparse, hashlib, shutil, glob
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional, Tuple

import yaml
import s3fs
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import bigquery
from google.cloud import bigquery_storage
from google.cloud import storage as gcs_storage
from google.cloud.storage import transfer_manager
from loguru import logger

from genml_training_tools.gcp.credentials import get_default_credentials

#---------------------------------------------------------------------------
# Small helpers to work with BQ/GCS/S3.

def get_bq_clients(project: str, location: str) -> Tuple[bigquery.Client, bigquery_storage.BigQueryReadClient]:
    creds, _ = get_default_credentials()
    return (
        bigquery.Client(credentials=creds, project=project, location=location),
        bigquery_storage.BigQueryReadClient(credentials=creds),
    )

def get_gcs_client(project: str) -> gcs_storage.Client:
    creds, _ = get_default_credentials()
    return gcs_storage.Client(credentials=creds, project=project)

def get_s3_filesystems(region: Optional[str]) -> Tuple[s3fs.S3FileSystem, pa.fs.S3FileSystem]:
    return s3fs.S3FileSystem(), pa.fs.S3FileSystem(region=region)

def unique_tmp_table_id(project: str, dataset: str, sql_query: str) -> str:
    salt = f"{time.time_ns()}:{os.getpid()}:{uuid4().hex}"
    h = hashlib.sha256((sql_query + salt).encode()).hexdigest()[:16]
    return f"{project}.{dataset}.tmp_{h}"

def ensure_temp_table_with_ttl(bq_client: bigquery.Client, table_id: str, ttl_hours: int = 6) -> None:
    table = bigquery.Table(table_id)
    table.expires = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
    try:
        bq_client.create_table(table)
    except Exception:
        pass  # benign

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
    return bq_client.list_rows(table, page_size=500_000).to_dataframe_iterable(bqstorage_client=bqstorage_client)

def _safe_put_to_s3(local_path: str, s3_path: str, cleanup: bool=False) -> None:
    size = os.path.getsize(local_path)
    min_parts = 8000
    part_size = max(8 * 1024 * 1024, math.ceil(size / min_parts))
    s3_uploader = s3fs.S3FileSystem(default_block_size=part_size)
    logger.info(f"Uploading {local_path} to {s3_path} with part size ~{part_size // (1024*1024)} MiB.")
    s3_uploader.put(local_path, s3_path)
    if cleanup:
        logger.info("Cleaning up local files...")
        os.remove(local_path)

def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    assert gs_uri.startswith("gs://"), f"Invalid GCS URI: {gs_uri}"
    parts = gs_uri[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix.rstrip("/")

def _extract_table_to_gcs_parquet(
    bq_client: bigquery.Client,
    table_id: str,
    gcs_folder: str,
) -> Tuple[str, str]:
    """
    Export Parquet shards to gs://bucket/prefix/<run_id>/*.parquet
    Returns (bucket, prefix_for_this_run)
    """
    run_id = uuid4().hex
    bucket, prefix = _parse_gs_uri(gcs_folder)
    export_prefix = f"{prefix}/{run_id}" if prefix else run_id
    dest_uri = f"gs://{bucket}/{export_prefix}/*.parquet"
    logger.info(f"Extracting {table_id} to {dest_uri} ...")
    job_config = bigquery.job.ExtractJobConfig(destination_format=bigquery.job.DestinationFormat.PARQUET)
    job = bq_client.extract_table(table_id, dest_uri, location=bq_client.location, job_config=job_config)
    job.result()
    logger.info("Extract job done.")
    return bucket, export_prefix

def _download_gcs_prefix_to_local(
    gcs_client: gcs_storage.Client,
    bucket: str,
    prefix: str,
    local_dir: str,
    max_workers: int = 64,
) -> None:
    os.makedirs(local_dir, exist_ok=True)
    bkt = gcs_client.bucket(bucket)
    blobs = [b.name for b in bkt.list_blobs(prefix=prefix) if b.name.endswith(".parquet")]
    logger.info(f"Downloading {len(blobs)} Parquet shards from gs://{bucket}/{prefix} to {local_dir} ...")
    transfer_manager.download_many_to_path(
        bkt,
        blobs,
        destination_directory=local_dir,
        max_workers=max_workers,
        worker_type=gcs_storage.transfer_manager.THREAD,
    )
    logger.info("Download complete.")

#---------------------------------------------------------------------------
# Parquet utils.

def build_parquet_from_chunks(
    dataframe_chunks_iterator: Iterator[pd.DataFrame],
    destination_path: str,
    val_destination_path: str,
    filesystem: pa.fs.FileSystem,
    total_rows: int,
    num_val_rows: int = 0,
) -> None:
    val_ratio = (num_val_rows / total_rows) if total_rows else 0.0
    assert val_ratio <= 0.5, f"num_val_rows ({num_val_rows}) must be < half of total rows ({total_rows})."

    writer = val_writer = None
    from tqdm import tqdm
    pbar = tqdm(total=total_rows, unit="rows", desc="Writing Parquet")

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

        writer.write_table(table_chunk)
        pbar.update(len(chunk_df))

    if writer:
        writer.close()

def _parquet_shards_dataframe_iter(shards_root_dir: str) -> Iterator[pd.DataFrame]:
    """
    Yield pandas DataFrames by scanning local Parquet shards (row-group by row-group).
    This lets us reuse build_parquet_from_chunks for merging and val splitting.
    """
    files = sorted(glob.glob(os.path.join(shards_root_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise RuntimeError(f"No Parquet shards found under {shards_root_dir}")
    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for i in range(pf.num_row_groups):
            tbl = pf.read_row_group(i)
            yield tbl.to_pandas(types_mapper=None)  # keep default dtype mapping

#---------------------------------------------------------------------------
# Orchestration

def construct_index_from_bq_query(
    bq_project: str,
    sql_query: str,
    s3_destination_path: str,
    recompute: bool = False,
    val_ratio: float = 0.1,
    max_num_val_rows: int = 10000,
    s3_bucket_region: Optional[str] = None,
    bq_dataset_for_temp: str = "generative_ai_data_platform_test",
    bq_location: str = "US",
    local_tmp_dir: Optional[str] = None,            # write locally then upload
    gcs_tmp_dir: Optional[str] = None,   # if provided (gs://...), use GCS export path
):
    assert s3_destination_path.startswith('s3://') and s3_destination_path.endswith(".parquet"), \
        f"Invalid S3 path: {s3_destination_path}"

    s3, s3_fs_pa = get_s3_filesystems(s3_bucket_region)
    val_s3_path = s3_destination_path.replace(".parquet", "-val.parquet")

    if s3.exists(s3_destination_path) and not recompute:
        logger.info(f"File already exists at {s3_destination_path}. Skipping computation.")
        return

    # Streaming path pre-compute (kept as-is)
    if local_tmp_dir:
        os.makedirs(local_tmp_dir, exist_ok=True)
        base_name = os.path.basename(s3_destination_path)
        base_val_name = os.path.basename(val_s3_path)
        dest_path_for_writer = os.path.join(local_tmp_dir, base_name)
        val_dest_for_writer = os.path.join(local_tmp_dir, base_val_name)
        pa_filesystem = pa.fs.LocalFileSystem()
        logger.info(f"Will write locally to {dest_path_for_writer} then upload to S3.")
    else:
        dest_path_for_writer = s3_destination_path.replace("s3://", "")
        val_dest_for_writer = val_s3_path.replace("s3://", "")
        pa_filesystem = s3_fs_pa

    bq_client, bqstorage_client = get_bq_clients(bq_project, bq_location)
    tmp_table = unique_tmp_table_id(bq_project, bq_dataset_for_temp, sql_query)

    ensure_temp_table_with_ttl(bq_client, tmp_table, ttl_hours=6)
    try:
        run_query_to_table(bq_client, sql_query, tmp_table)

        total_rows = table_row_count(bq_client, tmp_table)
        logger.info(f"Query finished. Found {total_rows} rows to process.")
        if total_rows == 0:
            logger.warning("Query returned 0 rows. No Parquet file will be created.")
            return

        # Compute val rows once so both paths behave the same
        if val_ratio == 0:
            num_val_rows = 0
            logger.warning("No validation set requested. This incident will be reported to Bobby.")
        else:
            num_val_rows = min(int(total_rows * val_ratio), max_num_val_rows)
            if num_val_rows > 100_000:
                logger.warning(f"Writing so many ({num_val_rows}) validation rows is too much. This incident will be reported to Evan.")
            logger.info(f"Will write {num_val_rows} validation rows.")

        if gcs_tmp_dir:
            # ---------- Large file size case: GCS Export + local merge using build_parquet_from_chunks. ----------
            gcs_client = get_gcs_client(bq_project)
            bucket, run_prefix = _extract_table_to_gcs_parquet(bq_client, tmp_table, gcs_tmp_dir)

            run_local_dir = os.path.join(local_tmp_dir or "/tmp", f"bq_export_{uuid4().hex}")
            shards_local_dir = os.path.join(run_local_dir, "shards")
            os.makedirs(shards_local_dir, exist_ok=True)

            _download_gcs_prefix_to_local(gcs_client, bucket, run_prefix, shards_local_dir)

            # Reuse existing writer by exposing shards as a DataFrame iterator
            merged_local_main = os.path.join(run_local_dir, os.path.basename(s3_destination_path))
            merged_local_val  = os.path.join(run_local_dir, os.path.basename(s3_destination_path).replace(".parquet", "-val.parquet"))

            df_iter = _parquet_shards_dataframe_iter(shards_local_dir)
            build_parquet_from_chunks(
                dataframe_chunks_iterator=df_iter,
                destination_path=merged_local_main,
                val_destination_path=merged_local_val,
                filesystem=pa.fs.LocalFileSystem(),
                total_rows=total_rows,
                num_val_rows=num_val_rows,
            )

            _safe_put_to_s3(merged_local_main, s3_destination_path, cleanup=True)
            if os.path.exists(merged_local_val) and os.path.getsize(merged_local_val) > 0 and num_val_rows > 0:
                _safe_put_to_s3(merged_local_val, val_s3_path, cleanup=True)

            shutil.rmtree(run_local_dir, ignore_errors=True)
            logger.info(f"Successfully wrote {total_rows} rows to {s3_destination_path} via GCS export.")
        else:
            # ---------- Streaming directly from BQ to local/S3. ----------
            df_iter = dataframe_iter_from_table(bq_client, bqstorage_client, tmp_table)
            build_parquet_from_chunks(
                dataframe_chunks_iterator=df_iter,
                destination_path=dest_path_for_writer,
                val_destination_path=val_dest_for_writer,
                filesystem=pa_filesystem,
                total_rows=total_rows,
                num_val_rows=num_val_rows,
            )

            if local_tmp_dir:
                _safe_put_to_s3(dest_path_for_writer, s3_destination_path, cleanup=True)
                if num_val_rows > 0 and os.path.exists(val_dest_for_writer):
                    _safe_put_to_s3(val_dest_for_writer, val_s3_path, cleanup=True)

            logger.info(f"Successfully wrote {total_rows} rows to {s3_destination_path}")
    finally:
        try:
            bq_client.delete_table(tmp_table, not_found_ok=True)
            logger.info(f"Deleted temp table {tmp_table}.")
        except Exception as e:
            logger.warning(f"Failed to delete temp table {tmp_table}: {e}")


#---------------------------------------------------------------------------
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
        bq_dataset_for_temp=cfg.get('bq_dataset_for_temp', 'generative_ai_data_platform_test'),
        bq_location=cfg.get('bq_location', 'US'),
        local_tmp_dir=cfg.get('local_tmp_dir', None),
        gcs_tmp_dir=cfg.get('gcs_tmp_dir'),
    )

#---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#---------------------------------------------------------------------------
