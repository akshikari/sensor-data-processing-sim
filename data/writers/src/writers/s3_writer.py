from datetime import datetime, timezone
from io import BytesIO
import os
from collections.abc import Mapping
import logging
from typing import Any
from uuid import UUID, uuid4

import boto3
from botocore.config import Config
import pandas as pd

logger = logging.getLogger(__name__)


# TODO: Maybe consider a transformers library to prepare data for more generic writer.
class S3Writer:
    """
    Takes in a list of dict ojbects, converts them to Parquet, and writes Parquet files and objects to S3
    """

    def __init__(
        self,
        bucket_name: str,
        base_s3_prefix: str,
        partition_columns: list[str],
        file_type: str = "csv",
        aws_region: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        assume_role_arn: str | None = None,
        role_session_name: str = "S3WriterSession",
        parquet_compression: str = "snappy",
    ):
        """
        Initialize S3 Writer

        :param bucket_name: Name of the S3 bucket to write the data to
        :param base_s3_prefix: Base "directory" path within the bucket.
        :param partition_columns: List of data attributes from which to generate the S3 partition key
        :param file_type: File type to write data as in S3 bucket (Parquet, CSV)
        :param aws_region: AWS region. If None, relies on env/config.
        :param aws_access_key_id: Optional explicit credential.
        :param aws_secret_access_key: Optional explicit credential.
        :param assume_role_arn: Optional IAM Role ARN to assume for credentials.
        :param role_session_name: Session name to use when assuming role.
        :param parquet_compression: Compression for Parquet ('snappy, 'gzip', None).
        :returns: AN S3Writer object with the ability to take a list of dict objects, serialize them to a specified file format, and write them to S3.
        :raises Exception: Will raise an exception if it fails to connect to the S3 instance
        """

        if not bucket_name:
            raise ValueError("bucket_name not specified")

        self.bucket_name: str = bucket_name
        self.base_s3_prefix: str = base_s3_prefix if base_s3_prefix else ""
        self.partition_columns: list[str] = partition_columns or []
        self.file_type: str = file_type

        self._aws_config: dict[str, str | None] = {
            "region_name": aws_region
            or os.environ.get("AWS_DEFAULT_REGION", "us-east-2"),
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "assume_role_arn": assume_role_arn,
            "role_session_name": role_session_name,
        }

        self.s3_client = self._build_client()
        logger.info(f"Initialized boto3 S3 client for bucket: {self.bucket_name}")

    def _build_client(self):
        """Creates the boto3 S3 client, handling optional explicit creds or role assumption."""
        config = Config(
            retries={"max_attempts": 5, "mode": "standard"},
            region_name=self._aws_config["region_name"],
        )

        # Connection setup priority: Assume Role -> Default Chain (env)
        if (
            not self._aws_config["aws_access_key_id"]
            and not self._aws_config["aws_secret_access_key"]
            and self._aws_config["assume_role_arn"]
        ):
            logger.debug(
                f"Attempting to assume role: {self._aws_config['assume_role_arn']}"
            )
            try:
                role_session_name = self._aws_config.get(
                    "role_session_name", "sensor-data-sim"
                )
                sts_session = boto3.Session()
                sts_connection = sts_session.client("sts")
                assume_role_object = sts_connection.assume_role(
                    RoleArn=self._aws_config["assume_role_arn"],
                    RoleSessionName=role_session_name,
                )

                session = boto3.Session(
                    aws_access_key_id=assume_role_object["Credentials"]["AccessKeyId"],
                    aws_secret_access_key=assume_role_object["Credentials"][
                        "SecretAccessKey"
                    ],
                    aws_session_token=assume_role_object["Credentials"]["SessionToken"],
                )

                return session.client(
                    "s3", config=config, region_name=self._aws_config["region_name"]
                )
            except Exception:
                logger.exception(
                    f"Failed to assume role {self._aws_config['assume_role_arn']}. Falling back to default credentials."
                )

        logger.debug("Using default AWS credential chain.")
        return boto3.client(
            "s3",
            region_name=self._aws_config["region_name"],
            config=config,
        )

    def _get_partition_path(self, record_dict: Mapping[str, Any]) -> str | None:
        """
        Generate Hive-style partition path string from record's data dictionary.

        Assumptions:
        - If passing in dates, assume it's already been partitioned into desired specificity e.g. year={year}, month={month}, etc.
        """
        # TODO: Move this to a transformer
        try:
            parts: list[str] = []
            for column in self.partition_columns:
                value = record_dict.get(column)
                if value is None:
                    logger.error(f"Missing value for partition column '{column}'.")
                    return None
                # Convert common types to string for path safety
                if isinstance(value, UUID):
                    value = str(value)
                parts.append(f"{column}={value}")
            return "/".join(parts)
        except Exception as err:
            logger.exception("Error generating partition path: %s", err)
            return None

    def write_batch(self, data_batch: list[dict[str, Any]]) -> None:
        """
        Writes a batch of Pydantic model objects to a partitioned Parquet file in AWS S3.

        Assumptions:
        - Input is a list of dictionaries with consistent keys.
        - All records in the batch belong to the same logical partition as determined by the data in the first record
        - Data contains keys corresponding to `partition_columns`

        :param data_batch: List of objects to be written to the self.bucket_name S3 Bucket under the self.partition_path
        """
        if not data_batch:
            logger.info("No data provided, no write operations performed.")
            return

        try:
            # TODO: Move this to transformers package
            # Determine partition path from the first record
            partition_path: str | None = self._get_partition_path(data_batch[0])
            if partition_path is None:
                raise ValueError(
                    "Could not determine the partition path for the batch."
                )

            df = pd.DataFrame.from_records(data_batch)

            buffer = BytesIO()
            if self.file_type == "parquet":
                df.to_parquet(
                    buffer,
                    allow_truncated_timestamps=True,
                )
            elif self.file_type == "csv":
                df.to_csv(buffer, index=False)
            else:
                raise Exception(f"File type {self.file_type} is not supported.")

            buffer.seek(0)

            curr_time = datetime.now(timezone.utc)
            file_id = uuid4()
            filename = (
                f"{file_id}_{curr_time.strftime('%Y%m%d-%H%M%S')}.{self.file_type}"
            )
            s3_object_key = "/".join(
                [self.base_s3_prefix.rstrip("/").lstrip("/"), partition_path, filename]
            )

            self.s3_client.put_object(
                Bucket=self.bucket_name, Key=s3_object_key, Body=buffer.getvalue()
            )

            logger.info("Successfully wrotefile to S3")

        except Exception as err:
            logger.exception("Failed to write batch to S3:\n%s", err)
            raise
