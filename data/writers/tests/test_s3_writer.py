from collections.abc import Iterator
import logging
import pytest
import os
import pandas as pd
import boto3
from moto import mock_aws
from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import Any, TYPE_CHECKING
from io import BytesIO, StringIO

# --- Type Hinting Setup ---
if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
    from pytest import LogCaptureFixture
else:
    S3Client = Any
    LogCaptureFixture = Any
# --------------------------

from writers.s3_writer import S3Writer

# --- Test Constants ---
TEST_BUCKET_NAME: str = "test-sensor-data-bucket"
TEST_BASE_PREFIX: str = "raw_data/mock_sensor/"
TEST_PARTITION_COLS: list[str] = ["year", "month", "day", "hour", "sensor_id"]


# --- Test Helper Functions ---
def create_test_dict(ts: datetime, sensor_id: UUID, val: float) -> dict[str, Any]:
    """Creates a dictionary with data and derived partition keys."""
    ts_utc = (
        ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    )
    sid_str = str(sensor_id)
    return {
        "timestamp": ts_utc,
        "sensor_id": sid_str,
        "accel_x": val,
        "accel_y": val + 0.1,
        "accel_z": val + 0.2,
        # Add derived partition keys matching TEST_PARTITION_COLS
        "year": ts_utc.year,
        "month": f"{ts_utc.month:02d}",
        "day": f"{ts_utc.day:02d}",
        "hour": f"{ts_utc.hour:02d}",
    }


def get_expected_partition_path(
    record_dict: dict[str, Any], partition_cols: list[str]
) -> str:
    """Calculates the expected Hive partition path string FOR TESTS."""
    parts = []
    for col in partition_cols:
        value = record_dict.get(col)
        if value is None:
            raise ValueError(f"Test data missing partition column for path: {col}")
        # Use str() for consistency in path generation, matching the writer's implicit str conversion
        parts.append(f"{col}={str(value)}")
    return "/".join(parts)


# --- Pytest Fixtures ---


@pytest.fixture(scope="function")
def aws_credentials() -> None:
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECRET_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-2"


@pytest.fixture(scope="function")
def s3_moto_client(aws_credentials: None) -> Iterator[S3Client]:
    with mock_aws():
        client: S3Client = boto3.client(
            "s3", region_name=os.environ["AWS_DEFAULT_REGION"]
        )
        yield client


@pytest.fixture(scope="function")
def s3_bucket(s3_moto_client: S3Client) -> str:
    s3_moto_client.create_bucket(
        Bucket=TEST_BUCKET_NAME,
        CreateBucketConfiguration={
            "LocationConstraint": os.environ["AWS_DEFAULT_REGION"]
        },
    )
    return TEST_BUCKET_NAME


# Fixture for Parquet writer instance
@pytest.fixture(scope="function")
def s3_parquet_writer(s3_bucket: str) -> S3Writer:
    return S3Writer(
        bucket_name=s3_bucket,
        base_s3_prefix=TEST_BASE_PREFIX,
        partition_columns=TEST_PARTITION_COLS,
        file_type="parquet",
        aws_region=os.environ["AWS_DEFAULT_REGION"],
    )


# Fixture for CSV writer instance
@pytest.fixture(scope="function")
def s3_csv_writer(s3_bucket: str) -> S3Writer:
    return S3Writer(
        bucket_name=s3_bucket,
        base_s3_prefix=TEST_BASE_PREFIX,
        partition_columns=TEST_PARTITION_COLS,
        file_type="csv",
        aws_region=os.environ["AWS_DEFAULT_REGION"],
    )


# --- Test Class ---
@mock_aws
class TestS3WriterBoto3:
    def test_init_success(self, s3_bucket: str):
        """Test successful initialization and client creation."""
        writer = S3Writer(
            bucket_name=s3_bucket,
            base_s3_prefix=TEST_BASE_PREFIX,
            partition_columns=TEST_PARTITION_COLS,
            file_type="parquet",
        )
        assert writer.bucket_name == s3_bucket
        assert writer.base_s3_prefix == TEST_BASE_PREFIX
        assert writer.partition_columns == TEST_PARTITION_COLS
        assert writer.file_type == "parquet"
        assert writer.s3_client is not None
        # Check if client is functional (moto handles this) - test_connection method was for this
        try:
            writer.s3_client.head_bucket(Bucket=s3_bucket)
        except Exception as e:
            pytest.fail(f"Mocked S3 client failed head_bucket: {e}")

    def test_init_invalid_bucket(self):
        """Test initialization fails with empty bucket name."""
        with pytest.raises(ValueError, match="bucket_name not specified"):
            _ = S3Writer(bucket_name="", base_s3_prefix="test", partition_columns=[])

    # Test the revised _get_partition_path
    def test_get_partition_path(
        self, s3_parquet_writer: S3Writer
    ):  # Use parquet writer fixture
        """Test partition path generation helper assuming pre-derived keys."""
        test_id = uuid4()
        record_dict = {
            "year": 2025,
            "month": "04",
            "day": "22",
            "hour": "15",
            "sensor_id": str(test_id),
            "other_data": 123,
        }
        # Adjust expected path based on TEST_PARTITION_COLS definition at top
        expected_path = f"year=2025/month=04/day=22/hour=15/sensor_id={test_id}"
        # Change partition cols for the test instance if needed, or adjust expected path
        s3_parquet_writer.partition_columns = [
            "year",
            "month",
            "day",
            "hour",
            "sensor_id",
        ]
        assert s3_parquet_writer._get_partition_path(record_dict) == expected_path  # noqa: SLF001

    def test_get_partition_path_missing_col(self, s3_parquet_writer: S3Writer):
        """Test partition path returns None if a required column is missing."""
        record_dict: dict[str, Any] = {"year": 2025, "month": "04"}
        s3_parquet_writer.partition_columns = ["year", "month", "sensor_id"]
        assert s3_parquet_writer._get_partition_path(record_dict) is None  # noqa: SLF001

    def test_write_batch_empty(
        self, s3_parquet_writer: S3Writer, s3_bucket: str, caplog: LogCaptureFixture
    ):
        """Test writing an empty batch does nothing and logs info."""
        caplog.set_level(logging.INFO)
        empty_batch: list[dict[str, Any]] = []
        s3_parquet_writer.write_batch(empty_batch)
        assert "No data provided, no write operations performed." in caplog.text
        s3_client_fixture = boto3.client("s3")
        response = s3_client_fixture.list_objects_v2(
            Bucket=s3_bucket, Prefix=s3_parquet_writer.base_s3_prefix
        )
        assert "Contents" not in response  # No objects should exist

    def test_write_batch_success_parquet(
        self, s3_parquet_writer: S3Writer, s3_bucket: str, s3_moto_client: S3Client
    ):
        """Test successful Parquet write and verify content."""
        test_id = uuid4()
        ts1 = datetime(2025, 4, 22, 15, 0, 1, tzinfo=timezone.utc)
        ts2 = datetime(2025, 4, 22, 15, 0, 2, tzinfo=timezone.utc)

        batch_data = [
            create_test_dict(ts1, test_id, 1.0),
            create_test_dict(ts2, test_id, 1.1),
        ]

        s3_parquet_writer.write_batch(batch_data)

        expected_partition_path = (
            f"year=2025/month=04/day=22/hour=15/sensor_id={test_id}"
        )
        expected_prefix = (
            f"{s3_parquet_writer.base_s3_prefix}{expected_partition_path}/"
        )

        response = s3_moto_client.list_objects_v2(
            Bucket=s3_bucket, Prefix=expected_prefix
        )
        assert "Contents" in response, f"No files found in {expected_prefix}"
        assert len(response["Contents"]) > 0
        written_s3_key = response["Contents"][0].get("Key")
        assert written_s3_key is not None
        assert written_s3_key.endswith(".parquet")

        try:
            s3_object = s3_moto_client.get_object(Bucket=s3_bucket, Key=written_s3_key)
            parquet_bytes = s3_object["Body"].read()
            df_read = pd.read_parquet(BytesIO(parquet_bytes))
        except Exception as e:
            pytest.fail(
                f"Failed to read back Parquet file '{written_s3_key}' from mocked S3: {e}"
            )

        assert len(df_read) == len(batch_data)
        pd.testing.assert_frame_equal(
            df_read.sort_index(axis=1),
            pd.DataFrame.from_records(batch_data).sort_index(axis=1),
            check_dtype=False,
            check_datetimelike_compat=True,
            atol=1e-6,
        )

    def test_write_batch_success_csv(
        self, s3_csv_writer: S3Writer, s3_bucket: str, s3_moto_client: S3Client
    ):
        """Test successful CSV write and verify content."""
        test_id = uuid4()
        ts1 = datetime(2025, 5, 1, 10, 0, 1, tzinfo=timezone.utc)

        batch_data = [create_test_dict(ts1, test_id, 5.5)]

        s3_csv_writer.write_batch(batch_data)

        expected_partition_path = (
            f"year=2025/month=05/day=01/hour=10/sensor_id={test_id}"
        )
        expected_prefix = f"{s3_csv_writer.base_s3_prefix}{expected_partition_path}/"

        response = s3_moto_client.list_objects_v2(
            Bucket=s3_bucket, Prefix=expected_prefix
        )
        assert "Contents" in response and len(response["Contents"]) > 0
        written_s3_key = response["Contents"][0].get("Key")
        assert written_s3_key is not None
        assert written_s3_key.endswith(".csv")

        try:
            s3_object = s3_moto_client.get_object(Bucket=s3_bucket, Key=written_s3_key)
            csv_bytes = s3_object["Body"].read()
            df_read = pd.read_csv(StringIO(csv_bytes.decode("utf-8")))
        except Exception as e:
            pytest.fail(
                f"Failed to read back CSV file '{written_s3_key}' from mocked S3: {e}"
            )

        assert len(df_read) == len(batch_data)
        assert df_read.iloc[0]["accel_x"] == pytest.approx(5.5)
        assert df_read.iloc[0]["sensor_id"] == str(test_id)
        assert pd.to_datetime(df_read.iloc[0]["timestamp"]).to_pydatetime() == ts1
        assert df_read.iloc[0]["year"] == ts1.year
        assert df_read.iloc[0]["month"] == ts1.month
        assert df_read.iloc[0]["day"] == ts1.day
        assert df_read.iloc[0]["hour"] == ts1.hour

    def test_write_batch_unsupported_type(self, s3_bucket: str):
        """Test that an unsupported file_type raises an exception."""
        writer = S3Writer(
            bucket_name=s3_bucket,
            base_s3_prefix=TEST_BASE_PREFIX,
            partition_columns=[],
            file_type="unsupported",
        )
        batch_data = [{"col1": 1}]
        with pytest.raises(Exception, match="File type unsupported is not supported"):
            writer.write_batch(batch_data)
