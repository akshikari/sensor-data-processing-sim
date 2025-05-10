"""Taking the hacky approach with this one. Want to learn other stuff."""

import math
import sys
import logging
from typing import Any
from datetime import datetime, timezone
from dataclasses import dataclass, field

import pandas as pd

from generators.accelerometer import (
    GenerateDataParams,
    generate_data,
)
from writers import S3Writer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfigs:
    source: dict[str, Any] = field(default_factory=dict)
    destination: dict[str, Any] = field(default_factory=dict)


def run_pipeline(configs: PipelineConfigs) -> bool:
    """
    Runs Pipeline to load generated accelerometer data to S3

    :param configs: Configuration dictionary providing the steps of the pipeline (e.g. source, transformations, and destination)
    :raises Exception: Propagates exceptions from generator or writer on failure, or if provided configuration is invalid
    """
    # HACK:
    # TODO: Write more generic pipeline builder that builds based off of JSON configs
    try:
        source_configs = configs.source
        destination_configs = configs.destination
        if not source_configs or not destination_configs:
            print(source_configs)
            print(destination_configs)
            logging.warning("Source and/or Destination configurations not provided")
            return False

        # Just putting this here to have the idea of validating pipeline components
        source_type: str = source_configs.get("type", "generator-accelerometer")
        if source_type != "generator-accelerometer":
            raise Exception("")

        logging.info("Generating data...")
        source_params: GenerateDataParams = GenerateDataParams(
            **source_configs.get("generator_configs", {})
        )
        frequency: int | float = configs.source.get("frequency", 100)
        total_time: int | float = configs.source.get("total_time", 10)
        start_ts: datetime = datetime.now(timezone.utc)
        data_df: pd.DataFrame = generate_data(
            frequency=frequency,
            total_time=total_time,
            start_time=start_ts,
            params=source_params,
        )
        if data_df.empty:
            logging.warning("No data generated")
            return False
        logging.info(f"Generated {len(data_df)} record(s).")

        # TODO: Write transformers to do the below steps
        logging.info(
            "Applying specified transformations: %s",
            ", ".join(["Mappings, GeneratePartitionPathFields"]),
        )  # logging statement will eventually draw and list actual transformer functions being used

        data_df["year"] = data_df["timestamp"].dt.year.astype(str)
        data_df["month"] = data_df["timestamp"].dt.strftime("%m")
        data_df["day"] = data_df["timestamp"].dt.strftime("%d")
        data_df["hour"] = data_df["timestamp"].dt.strftime("%H")

        logging.info(f"Applied transformations to {len(data_df)} record(s)")

        logging.info("Writing batch to destination...")
        writer = S3Writer(
            bucket_name=configs.destination.get("bucket_name", ""),
            base_s3_prefix=configs.destination.get("base_prefix", ""),
            partition_columns=["year", "month", "day", "hour", "sensor_id"],
            file_type=configs.destination.get("file_type", "parquet"),
        )
        writer.write_batch(data_df)
        # HACK: END

        logger.info("Pipeline completed successfully.")
        return True
    except Exception as err:
        # TODO: Tighten up error handling
        logging.error("Pipeline failed due to the following error:\n%s", err)
        raise


if __name__ == "__main__":
    source_configs = {
        "type": "generator-accelerometer",
        "frequency": 100,
        "total_time": 10,
        "generator_configs": {
            "gait_frequency": 2.1,
            "speed": 1.2,
            "amplitude_sway": 0.04,
            "amplitude_bounce": 0.025,
            "amplitude_roll": 0.06,
            "amplitude_pitch": 0.04,
            "base_pitch": 0.01,
            "base_roll": 0.0,
            "noise_std_dev": 0.03,
            "phase_sway": 0.0,
            "phase_bounce": math.pi / 2,
            "phase_roll": math.pi,
            "phase_pitch": 0.0,
        },
    }
    destination_configs = {
        "type": "s3",
        "bucket_name": "sensor-sim-raw-accelerometer-data20250501160428685300000001",
        "base_prefix": "raw_data/accelerometer",
        "partition_columns": ["year", "month", "day", "hour", "sensor_id"],
        "file_type": "parquet",
    }
    configs = PipelineConfigs(source=source_configs, destination=destination_configs)
    try:
        logger.info(
            "Configuration loaded successfully. Starting pipeline {pipeline name/id}..."
        )
        success = run_pipeline(configs)
    except Exception as err:
        logger.exception(
            "An unexpected error occurred during pipeline execution:\n%s", err
        )
        sys.exit(1)
