import logging
from uuid import uuid4
from unittest.mock import MagicMock
from datetime import datetime, timezone, timedelta

from writers.s3_writer import S3Writer
from streams.pipe_accelerometer_to_s3 import run_pipeline, PipelineConfigs
from generators.accelerometer import AccelerometerData, GenerateDataParams

import pandas as pd
import pytest
from pytest_mock import MockerFixture

# Constants
TEST_BUCKET = "mock-bucket"
TEST_PREFIX = "mock/prefix"
DEFAULT_PARTITION_COLUMNS = ["year", "month", "day", "hour", "sensor_id"]


@pytest.fixture
def default_pipeline_config() -> PipelineConfigs:
    """Provides a default PipelineConfigs object for tests."""
    return PipelineConfigs(
        source={
            "type": "generator-accelerometer",
            "frequency": 10,
            "total_time": 1,
            "generator_configs": {"noise_std_dev": 0.01},
        },
        destination={
            "type": "s3",
            "bucket_name": TEST_BUCKET,
            "base_prefix": TEST_PREFIX,
            "partition_columns": DEFAULT_PARTITION_COLUMNS,
            "file_type": "parquet",
        },
    )


@pytest.fixture(scope="function")
def mock_dependencies(mocker: MockerFixture) -> dict[str, MagicMock]:
    """
    Mocks generate_data and S3Writer, returning the mocks in a dictionary.
    Uses the mocker fixture provided by pytest-mock.
    """
    mock_gen_default_resp = pd.DataFrame()  # default return value
    mock_generator = mocker.patch(
        "streams.pipe_accelerometer_to_s3.generate_data",
        return_value=mock_gen_default_resp,
    )

    mock_s3writer_class = mocker.patch(
        "streams.pipe_accelerometer_to_s3.S3Writer", spec=S3Writer
    )
    mock_writer_instance = MagicMock(spec=S3Writer)
    mock_s3writer_class.return_value = mock_writer_instance

    return {
        "generate_data": mock_generator,
        "S3Writer": mock_s3writer_class,
        "s3_writer_instance": mock_writer_instance,
    }


def create_dummy_data(count: int) -> pd.DataFrame:
    """Creates a list of simple AccelerometerData objects"""
    if count == 0:
        return pd.DataFrame(
            columns=["timestamp", "sensor_id", "accel_x", "accel_y", "accel_z"]
        )

    now = datetime.now(timezone.utc)
    sensor_id = uuid4()

    data_dicts: list[AccelerometerData] = [
        AccelerometerData(
            timestamp=(now + timedelta(seconds=i * 0.1)).replace(tzinfo=timezone.utc),
            sensor_id=sensor_id,
            accel_x=float(i),
            accel_y=float(i) + 0.1,
            accel_z=float(i) + 0.2,
        )
        for i in range(count)
    ]

    df = pd.DataFrame.from_records(data_dicts)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["sensor_id"] = df["sensor_id"].astype(str)
    return df


class TestRunPipeline:
    def test_successful_run(
        self,
        default_pipeline_config: PipelineConfigs,
        mock_dependencies: dict[str, MagicMock],
        caplog: pytest.LogCaptureFixture,
    ):
        """Test a successful run of the pipeline using mocks"""
        caplog.set_level(logging.INFO)

        mock_data = create_dummy_data(10)
        mock_dependencies["generate_data"].return_value = mock_data

        success = run_pipeline(default_pipeline_config)

        # Function call asssertions
        assert success is True
        mock_dependencies["generate_data"].assert_called_once()
        _, call_kwargs = mock_dependencies["generate_data"].call_args
        assert call_kwargs.get("frequency") == 10
        assert call_kwargs.get("total_time") == 1
        assert isinstance(call_kwargs.get("params"), GenerateDataParams)

        mock_dependencies["S3Writer"].assert_called_once_with(
            bucket_name=TEST_BUCKET,
            base_s3_prefix=TEST_PREFIX,
            partition_columns=["year", "month", "day", "hour", "sensor_id"],
            file_type="parquet",
        )

        # Data quality assertions
        mock_dependencies["s3_writer_instance"].write_batch.assert_called_once()
        written_df = mock_dependencies["s3_writer_instance"].write_batch.call_args[0][0]
        assert isinstance(written_df, pd.DataFrame)
        assert len(written_df) == 10
        assert "year" in written_df.columns
        assert "month" in written_df.columns
        assert "day" in written_df.columns
        assert "hour" in written_df.columns
        assert "sensor_id" in written_df.columns
        assert pd.api.types.is_string_dtype(written_df["sensor_id"])

        assert "Pipeline completed successfully" in caplog.text

    def test_no_data_generated(
        self,
        default_pipeline_config: PipelineConfigs,
        mock_dependencies: dict[str, MagicMock],
        caplog: pytest.LogCaptureFixture,
    ):
        """Test the case where the generator returns an empty list."""
        caplog.set_level(logging.INFO)

        success = run_pipeline(default_pipeline_config)

        # Function call assertions
        assert success is False
        mock_dependencies["generate_data"].assert_called_once()
        # Writer should NOT have been initialized or called
        mock_dependencies["S3Writer"].assert_not_called()
        mock_dependencies["s3_writer_instance"].write_batch.assert_not_called()
        assert "No data generated" in caplog.text  # Check log message

    def test_writer_init_fails(
        self,
        default_pipeline_config: PipelineConfigs,
        mock_dependencies: dict[str, MagicMock],
        caplog: pytest.LogCaptureFixture,
    ):
        """Test pipeline failure if S3Writer initialization fails."""
        caplog.set_level(logging.INFO)

        mock_dependencies["generate_data"].return_value = create_dummy_data(1)
        mock_dependencies["S3Writer"].side_effect = ConnectionError(
            "Mock S3 Conn Error"
        )

        with pytest.raises(ConnectionError, match="Mock S3 Conn Error"):
            run_pipeline(default_pipeline_config)

        assert "Pipeline failed" in caplog.text
        assert "Mock S3 Conn Error" in caplog.text
        mock_dependencies["s3_writer_instance"].write_batch.assert_not_called()

    def test_writer_write_batch_fails(
        self,
        default_pipeline_config: PipelineConfigs,
        mock_dependencies: dict[str, MagicMock],
        caplog: pytest.LogCaptureFixture,
    ):
        """Test pipeline failure if S3Writer.write_batch fails."""
        caplog.set_level(logging.INFO)

        mock_dependencies["generate_data"].return_value = create_dummy_data(1)
        mock_writer_instance = mock_dependencies["s3_writer_instance"]
        mock_dependencies["s3_writer_instance"].write_batch.side_effect = IOError(
            "Mock S3 Write Error"
        )

        # Expect run_pipeline to re-raise the exception
        with pytest.raises(IOError, match="Mock S3 Write Error"):
            run_pipeline(default_pipeline_config)

        # Check logs
        mock_dependencies["S3Writer"].assert_called_once()  # Writer class was called
        mock_writer_instance.write_batch.assert_called_once()  # write_batch was called
        assert "Pipeline failed" in caplog.text
        assert "Mock S3 Write Error" in caplog.text

    def test_missing_config(self, caplog: pytest.LogCaptureFixture):
        """Test pipeline returns False if config is missing source or destination."""
        caplog.set_level(logging.INFO)
        config_no_dest = PipelineConfigs(source={"type": "generator-accelerometer"})
        success1 = run_pipeline(config_no_dest)
        assert success1 is False
        assert "Source and/or Destination configurations not provided" in caplog.text

        caplog.clear()
        config_no_src = PipelineConfigs(destination={"type": "s3", "bucket_name": "b"})
        success2 = run_pipeline(config_no_src)
        assert success2 is False
        assert "Source and/or Destination configurations not provided" in caplog.text

    def test_invalid_source_type(self, caplog: pytest.LogCaptureFixture):
        """Test pipeline raises error for unsupported source type."""
        caplog.set_level(logging.INFO)
        invalid_config = PipelineConfigs(
            source={"type": "invalid-generator"},
            destination={"type": "s3", "bucket_name": "b"},
        )
        with pytest.raises(Exception):
            run_pipeline(invalid_config)
        assert "Pipeline failed" in caplog.text
