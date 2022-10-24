from mmcv.runner.hooks.hook import Hook, HOOKS
import os
from pathlib import Path
from typing import Dict, Optional, Union
import boto3
import botocore
import torch.distributed as dist
import logging
from urllib.parse import parse_qs, urlparse

WAABI_OUTPUT_DIR_ENV = "WAABI_OUTPUT_DIR"
WAABI_S3_JOB_FOLDER_ENV = "WAABI_S3_JOB_FOLDER"
WAABI_USER_ENV = "WAABI_USER"
WAABI_S3_DOWNLOAD_ENV = "WAABI_S3_DOWNLOAD"
WAABI_JOB_NUM_PROC_ENV = "WAABI_JOB_NUM_PROC"
JOB_DATA_S3_BUCKET_ENV = "JOB_DATA_S3_BUCKET"
WAABI_BATCH_JOB_ARRAY_SIZE_ENV = "WAABI_BATCH_JOB_ARRAY_SIZE"
AWS_BATCH_JOB_ARRAY_INDEX_ENV = "AWS_BATCH_JOB_ARRAY_INDEX"
_logger = logging.getLogger(__name__)


def extract_prefix_and_bucket_from_uri(s3_prefix_uri: str):
    """Get bucket and prefix from the URI"""
    parsed_uri = urlparse(s3_prefix_uri)
    assert parsed_uri.scheme == "s3"
    bucket = parsed_uri.netloc
    prefix = parsed_uri.path.strip("/")
    # don't append "/" if prefix is empty
    # otherwise "/" gets treated as a "folder" on s3
    if len(prefix) > 0:
        prefix += "/"
    return bucket, prefix


def aws_s3_upload_dir(
    local_dir: Union[Path, str],
    s3_prefix_uri: str,
    *,
    s3_client=None,
    skip_existing_objects: bool = False,
    verbose: bool = False,
):
    if s3_client is None:
        s3_client = boto3.client("s3")

    bucket, prefix = extract_prefix_and_bucket_from_uri(s3_prefix_uri)

    base_path = Path(local_dir)  # support str input
    assert base_path.is_dir()

    skip_keys = dict()
    if skip_existing_objects:
        pages = s3_client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        for list_objects_response in pages:
            for obj in list_objects_response["Contents"]:
                skip_keys[obj["Key"]] = obj

    for dirpath, _, files in os.walk(base_path):
        for filename in files:
            # Skip an tmp files
            if ".tmp" in filename:
                continue
            # always use '/' separation in s3; do not use os path separator
            key = (Path(prefix) / Path(dirpath).relative_to(base_path) / filename).as_posix()
            if (
                key in skip_keys.keys()
                and Path(os.path.join(dirpath, filename)).lstat().st_size == skip_keys[key]["Size"]
            ):
                continue
            if verbose:
                _logger.info("uploading s3://%s/%s", bucket, key)
            s3_client.upload_file(Filename=os.path.join(dirpath, filename), Bucket=bucket, Key=key)


@HOOKS.register_module()
class SyncAWSHook(Hook):
    def __init__(self):
        print("invoke syncaws hook")

    def before_run(self, runner):
        print('before run, sync aws')
        if "WORLD_SIZE" in os.environ and dist.get_rank() == 0:
            sync_local_folder_to_s3()

    def after_epoch(self, runner):
        print('after epoch, sync aws', )
        if "WORLD_SIZE" in os.environ and dist.get_rank() == 0:
            sync_local_folder_to_s3()


def sync_local_folder_to_s3() -> bool:
    """Sync local folder to S3 based on env variables"""
    if WAABI_OUTPUT_DIR_ENV not in os.environ:
        return False

    if not os.path.exists(os.environ[WAABI_OUTPUT_DIR_ENV]):
        os.makedirs(os.environ[WAABI_OUTPUT_DIR_ENV])

    if WAABI_S3_JOB_FOLDER_ENV not in os.environ or not os.environ[WAABI_S3_JOB_FOLDER_ENV].startswith("s3://"):
        return False

    aws_s3_upload_dir(
        os.environ[WAABI_OUTPUT_DIR_ENV], os.environ[WAABI_S3_JOB_FOLDER_ENV], skip_existing_objects=True, verbose=False
    )
    return True
