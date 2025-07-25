"""
env/usage:
# pip install genml-training-tools -> Install this within the first hour of job start
# pip install --upgrade google-cloud
# pip install --upgrade google-cloud-bigquery
# pip install --upgrade google-cloud-storage
# clear && python prepare_bq_tables.py --config prepare_bq_tables.yaml
"""

import os
import time
import uuid
import shutil
import argparse
import yaml
import boto3
import botocore
from typing import List, Tuple, Union
from pydantic import BaseModel, field_validator
import os, uuid, shutil
from google.cloud import storage, bigquery
from google.cloud.storage import transfer_manager
from google.cloud.storage.bucket import Bucket
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import logging

from google.cloud import storage
from google.cloud import bigquery
from google.cloud.storage import transfer_manager
from genml_training_tools.gcp.credentials import get_default_credentials

logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)

#---------------------------------------------------------------------------

def prepare_bq_tables(
    s3_client,
    bq_project: str,
    filtration_query: str,
    filtered_table_folder: str,
    gcs_destination_root_dir: str,
    aws_destination_root_dir: str,
    metadata_tables: list[str] = None,
    metadata_columns: list[list[str | tuple]] = None,
    metadata_types: list[list[str]] = None,
    metadata_unique_keys: list[str] = None,
    recompute: bool=False,
):
    assert aws_destination_root_dir.startswith("s3://"), "aws_destination_root_dir must start with 's3://'"
    assert gcs_destination_root_dir.startswith("gs://"), "gcs_destination_root_dir must start with 'gs://'"
    assert aws_destination_root_dir.endswith("/"), "aws_destination_root_dir must end with '/'"
    assert gcs_destination_root_dir.endswith("/"), "gcs_destination_root_dir must end with '/'"
    print(f"Preparing data for BQ with {filtration_query = }, {filtered_table_folder = }")

    creds, _ = get_default_credentials()
    bq_client = bigquery.Client(credentials=creds, project=bq_project)
    storage_client = storage.Client(credentials=creds, project=bq_project)

    s3_bucket = aws_destination_root_dir.split("/")[2]
    s3_prefix = "/".join(aws_destination_root_dir.split("/")[3:])
    s3_split_path_key = os.path.join(s3_prefix.rstrip("/"), "split_file_paths.txt")

    if s3_key_exists(s3_client, s3_bucket, s3_split_path_key) and not recompute:
        print(f"s3://{s3_bucket}/{s3_split_path_key} already exists and `recompute` is False. Skipping recomputation.")
        return

    print("Recomputing the data...")

    # === Step 1: Filter table ===
    t0 = time.time()
    tmp_filtered_table = filter_table_and_save(
        bq_client,
        filtration_query,
        filtered_table_folder,
        metadata_tables,
        metadata_columns,
        metadata_types,
        metadata_unique_keys,
    )
    print(f"Filtering done in {time.time() - t0:.2f} sec")

    # === Step 2: Export to S3 ===
    t0 = time.time()
    _, split_file_paths = export_bq_to_s3(
        bq_client,
        storage_client,
        tmp_filtered_table,
        gcs_destination_root_dir,
        aws_destination_root_dir,
    )
    print(f"Exported {len(split_file_paths)} files in {time.time() - t0:.2f} sec")

    # === Step 3: Cleanup temp BQ table ===
    bq_client.delete_table(tmp_filtered_table)
    print(f"Deleted temporary table: {tmp_filtered_table}")

    # === Step 4: Upload file paths list ===
    upload_lines_to_s3(s3_client, split_file_paths, s3_bucket, s3_split_path_key)
    print(f"Uploaded split paths list to s3://{s3_bucket}/{s3_split_path_key}")

    return split_file_paths


def filter_table_and_save(
    bq_client,
    filtration_query: str,
    filtered_table_folder: str,
    metadata_tables: list[str] = None,
    metadata_columns: list[list[tuple]] = None,
    metadata_types: list[list[str]] = None,
    metadata_unique_keys: list[str] = None,
):
    tmp_id = uuid.uuid4().hex
    tmp_filtered_table = f"{filtered_table_folder}.data_filtered_{tmp_id}"

    print(f"Filtering data to: {tmp_filtered_table}")
    query_config = bigquery.QueryJobConfig(destination=tmp_filtered_table)
    bq_client.query(filtration_query, job_config=query_config).result()

    if metadata_tables:
        assert all([metadata_columns, metadata_types, metadata_unique_keys]), "Missing metadata config"

        for table, columns, types, key in zip(metadata_tables, metadata_columns, metadata_types, metadata_unique_keys):
            alter_sql = f"ALTER TABLE {tmp_filtered_table} {_generate_add_columns_sql(columns, types)}"
            update_sql = _generate_update_sql(tmp_filtered_table, table, columns, key)

            print(f"{alter_sql = }")
            print(f"{update_sql = }")

            bq_client.query(alter_sql).result()
            bq_client.query(update_sql).result()

    return tmp_filtered_table


def export_bq_to_s3(
    bq_client,
    storage_client,
    table_ref,
    gcs_base_uri,
    s3_base_uri,
    file_format="parquet",          # csv, parquet, etc.
):
    job_id = uuid.uuid4().hex
    gcs_uri = f"{gcs_base_uri.rstrip('/')}/{job_id}/*.{file_format}"
    local_tmp_dir_root = f"/tmp/{job_id}"
    os.makedirs(local_tmp_dir_root, exist_ok=True)

    # === EXPORT FROM BQ TO GCS ===
    extract_job_config = None
    if file_format == "parquet":
        extract_job_config = bigquery.job.ExtractJobConfig(destination_format="PARQUET")

    _extract_job_result = bq_client.extract_table(table_ref, gcs_uri, location="US", job_config=extract_job_config).result()

    # === FIGURE OUT THE SPLITS ===
    gcs_bucket_name: str = gcs_uri.split("/")[2] # E.g., `my-bucket`
    gcs_bucket: Bucket = storage_client.bucket(gcs_bucket_name)
    gcs_prefix = "/".join(gcs_uri.split("/")[3:]).replace(f"*.{file_format}", "")
    blobs = list(gcs_bucket.list_blobs(prefix=gcs_prefix))
    blob_paths = [blob.name for blob in blobs]
    print(f"Found {len(blob_paths)} blobs in the GCS bucket.")

    # === DOWNLOAD FROM GCS ===
    transfer_manager.download_many_to_path(gcs_bucket, blob_paths, destination_directory=local_tmp_dir_root, max_workers=16, worker_type=transfer_manager.THREAD)

    print(f'Downloaded data tables to {local_tmp_dir_root}. Will be uploading them now to S3...')

    # === UPLOAD TO S3 ===
    local_tmp_dir = os.path.join(local_tmp_dir_root, gcs_prefix)
    s3_target_uri = os.path.join(s3_base_uri, job_id)
    s3_full_file_paths = [os.path.join(s3_target_uri, f) for f in os.listdir(local_tmp_dir)]
    os.system(f"aws s3 cp {local_tmp_dir} {s3_target_uri} --recursive --quiet")

    print(f"Uploaded data tables to {s3_target_uri}! Deleting it locally now.")

    # === CLEANUP ===
    shutil.rmtree(local_tmp_dir_root)

    return job_id, s3_full_file_paths

#---------------------------------------------------------------------------
# S3 utils.

def s3_key_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise

def upload_lines_to_s3(s3_client, lines: list[str], bucket: str, key: str):
    local_tmp_path = f"/tmp/{uuid.uuid4().hex}_split_file_paths.txt"
    with open(local_tmp_path, "w") as f:
        f.writelines(f"{line}\n" for line in lines)

    with open(local_tmp_path, "rb") as f:
        s3_client.upload_fileobj(f, bucket, key)

    os.remove(local_tmp_path)

#---------------------------------------------------------------------------
# SQL utils.

def _generate_add_columns_sql(columns: list[tuple], types: list[str]) -> str:
    return ", ".join([f"ADD COLUMN {col[1]} {col_type}" for col, col_type in zip(columns, types)]) + ";"

def _generate_update_sql(filtered_table: str, metadata_table: str, columns: list[tuple], unique_key: str) -> str:
    set_clauses = ", ".join([f"data_table.{col[1]} = metadata_table.{col[0]}" for col in columns])
    select_cols = ", ".join([col[0] for col in columns])

    return f"""
    UPDATE {filtered_table} AS data_table
    SET {set_clauses}
    FROM (
        SELECT {unique_key}, {select_cols}
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY {unique_key}) AS rn
            FROM {metadata_table}
        )
        WHERE rn = 1
    ) AS metadata_table
    WHERE data_table.data_id = metadata_table.{unique_key};
    """

#---------------------------------------------------------------------------
# Config utils.

class ScriptConfig(BaseModel):
    bq_project: str
    filtration_query: str
    filtered_table_folder: str
    gcs_destination_root_dir: str = "gs://iskorokhodov/streaming-dl-bq-results/"
    aws_destination_root_dir: str
    metadata_tables: List[str] = []
    metadata_columns: List[List[Union[str, Tuple[str, ...]]]] = []
    # Field(
    #     ..., description="Nested list with strings and tuples"
    # )
    # metadata_columns features:
    #    1. It will be a List[List[str | tuple]]
    #    2. The length of the list should be equal to the number of metadata_tables we are working with i.e. each List[str | tuple] would correspond to one particular metadata_table in sequential order
    #    3. Eg. of List[tuple] where the tuple would be (the column name from the source metadata table, modified column name of choice in the merged table) to avoid mistakes like updating the same column names.
    #    4. Eg. of List[str] where the str would be the column name from the source metadata table. In this case, the column name in the merged table would be the same as the one in the metadata table.

    metadata_types: List[List[str]] = []
    metadata_unique_keys: List[str] = []

    @field_validator("metadata_columns", "metadata_types", mode="before")
    @classmethod
    def parse_nested_list(cls, value):
        """
        Ensures that tuples within the list are correctly converted from string representation.
        """
        if not isinstance(value, list):
            raise ValueError("list_of_list_mixed must be a list of lists")

        parsed_value = []
        for sublist in value:
            if not isinstance(sublist, list):
                raise ValueError("Each item in list_of_list_mixed must be a list")
            parsed_sublist = []
            for item in sublist:
                if isinstance(item, str) and item.startswith("(") and item.endswith(")"):
                    parsed_sublist.append(tuple(item.strip("()").split(",")))
                else:
                    parsed_sublist.append(item)
            parsed_value.append(parsed_sublist)

        return parsed_value


def load_yaml_config(file_path: str) -> ScriptConfig:
    """Reads the YAML file and validates it using Pydantic."""
    with open(file_path, "r") as file:
        raw_config = yaml.safe_load(file)

    # Validate and parse using Pydantic
    config = ScriptConfig(**raw_config)
    return config


def prepare_config(config):
    assert (
        len(config.metadata_tables)
        == len(config.metadata_columns)
        == len(config.metadata_types)
        == len(config.metadata_unique_keys)
    )

    if len(config.metadata_tables) == 0:
        return config

    new_metadata_columns = []
    for current_metadata_columns in config.metadata_columns:
        new_current_metadata_columns = []
        for metadata_column in current_metadata_columns:
            if isinstance(metadata_column, str):
                # Basically, we set the name of the column in the merged table to be the same as the name of the column in the metadata table
                metadata_column = (metadata_column, metadata_column)
            new_current_metadata_columns.append(metadata_column)
        new_metadata_columns.append(new_current_metadata_columns)
    config.metadata_columns = new_metadata_columns
    return config

#---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    configuration = load_yaml_config(args.config)
    configuration = prepare_config(configuration)
    s3_client = boto3.client("s3")

    prepare_bq_tables(
        s3_client,
        configuration.bq_project,
        configuration.filtration_query,
        configuration.filtered_table_folder,
        configuration.gcs_destination_root_dir,
        configuration.aws_destination_root_dir,
        configuration.metadata_tables,
        configuration.metadata_columns,
        configuration.metadata_types,
        configuration.metadata_unique_keys,
    )

#---------------------------------------------------------------------------
