import logging
from typing import Any
from datetime import datetime
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel
import s3fs

logger = logging.getLogger(__name__)


# TODO: If more file formats start to come through, reformat to more generic S3 writer.
# TODO: Maybe consider a transformers library to prepare data for more generic writer.
# TODO: Consider skipping pandas in converting to Parquet format. Fewer dependencies
class S3ParquetWriter:
    """
    Takes in Pydantic models, converts them to Parquet, and writes Parquet files and objects to S3

    Assumptions:
    - AWS S3 credentials are configured in the environment (fine for now with batch
    processes)
    """

    def __init__(
        self,
        bucket_name: str,
        base_s3_prefix: str,
        partition_columns: list[str],
        s3_conn_configs: dict[str, bool | int | str] | None = None,
        parquet_compression: str = "snappy",
    ):
        """
        Initialize S3 Writer

        :param bucket_name: Name of the S3 bucket to write the data to
        :param base_s3_prefix: Base "directory" path within the bucket.
        :param partition_columns: List of data attributes from which to generate the
        S3 partition key
        :param s3_conn_configs: Optional connection kwargs to pass to s3fs.S3FileSystem
        when establishing the connection
        :returns: AN S3 ParquetWriter object with the ability to take Pydantic model objects, serialize them to Parquet, and write to S3.
        :raises Exception: Will raise an exception if it fails to connect to the S3 instance
        """

        # Validations
        if not bucket_name:
            raise ValueError("bucket_name not specified")

        self.bucket_name: str = bucket_name
        self.base_s3_prefix: str = base_s3_prefix if base_s3_prefix else ""
        self.partition_columns: list[str] = partition_columns or []
        self.s3_conn_config: dict[str, bool | int | str] = s3_conn_configs or {}

        try:
            # Initialize S3 connection
            self.s3_fs = s3fs.S3FileSystem()
            logger.info("Initialized S3 AWS connection")
        except Exception as err:
            logger.exception(
                "Failed to initialize S3 AWS connection due to the following error:\n%s",
                err,
            )
            raise

    def _get_partition_path(self, record_dict: dict[str, Any]) -> str | None:
        """
        Generate Hive-style partition path string from record's data.

        Assumptions:
        - For now, assumes if partitioning with a timestamp, default to partitioning up to the hour
        -

        :param record_dict: A dictionary object representing a single
        :returns: String containing the Hive-style partition path or None if some error occurs
        """
        try:
            parts: list[str] = []
            for column in self.partition_columns:
                value = record_dict.get(column)
                if value is None:
                    logging.error("Missing value for generating partition key")
                    return None
                if isinstance(value, datetime):
                    parts.append(
                        *[
                            f"year={value.year}",
                            f"month={value.month:02d}",
                            f"day={value.day:02d}",
                            f"hour={value.hour:02d}",
                        ]
                    )
                else:
                    parts.append(f"{column}={value}")

            return "/".join(parts)
        except Exception as err:
            logger.error("Error generating partition path:\n%s", err)
            return None

    def write_batch(self, data_batch: list[BaseModel]) -> None:
        """
        Writes a batch of Pydantic model objects to a partitioned Parquet file in AWS S3.

        Assumptions:
        - All incoming objects belong to the same partition path as determined by the
        first element in the data batch
        - Defaults to snappy compression

        :param data_batch: List of objects (all of the same type T) to be written to
        the self.bucket_name S3 Bucket under the self.partition_path
        """
        if not data_batch:
            logger.info("No data provided, no write operations performed.")
            return

        try:
            # Convert Pydantic models to list of dictionaries
            record_dicts: list[dict[str, Any]] = [
                record.model_dump(mode="python") for record in data_batch
            ]

            # Generate s3 file path
            partition_path: str | None = self._get_partition_path(record_dicts[0])
            if partition_path is None:
                logger.error(
                    "Could not determine partition path for batch, skipping write"
                )
                return
            filename = f"{uuid4()}.parquet"
            s3_full_path = (
                f"{self.bucket_name}/{self.base_s3_prefix}/{partition_path}/{filename}"
            )

            # Convert data to Parquet format and write to S3
            records_df: pd.DataFrame = pd.DataFrame.from_records(record_dicts)
            records_df.to_parquet(
                f"s3://{s3_full_path}",
                engine="pyarrow",
                compression="snappy",
                index=False,
                filesystem=self.s3_fs,
            )
            logger.info("Successfully wrote Parquet file to S3")

        except Exception as err:
            logger.exception("Failed to write batch to S3:\n%s", err)
