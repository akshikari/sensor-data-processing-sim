from uuid import UUID
from generators.accelerometer import (
    AnomalousDataModifierParams,
    GenerateDataParams,
    generate_data,
)

import time
from datetime import datetime

import pytest
import numpy as np
import pandas as pd
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
from pydantic import ValidationError


TEST_CASES = {
    "invalidInputs": [
        pytest.param(None, None, ValidationError, id="null_input"),
        pytest.param(42, 0, ValidationError, id="zero_time"),
        pytest.param(0, 42, ValidationError, id="zero_frequency"),
        pytest.param(0, 0, ValidationError, id="zero_frequency_and_time"),
        pytest.param(42, -42, ValidationError, id="negative_time"),
        pytest.param(-42, 42, ValidationError, id="negative_frequency"),
        pytest.param(-42, -42, ValidationError, id="negative_frequency_and_time"),
        pytest.param(42.0, -42.0, ValidationError, id="negative_float_time"),
        pytest.param(-42.0, 42, ValidationError, id="negative_float_frequency"),
        pytest.param(-42.0, -42.0, ValidationError, id="negative_float_time_frequency"),
    ],
    "validInputsEmptyResult": [
        pytest.param(1.0, 0.5, id="floats_resulting_zero_samples_1"),
        pytest.param(1.99, 0.5, id="float_resulting_zero_samples_2"),
    ],
    "performanceTestCases": [
        pytest.param(100, 10, 1_000, id="1k_samples"),  # 1k samples
        pytest.param(100, 100, 10_000, id="10k_samples"),  # 10k samples
        pytest.param(1000, 100, 100_000, id="100k_samples"),  # 100k samples
        pytest.param(
            1000, 1000, 1_000_000, id="1M_samples", marks=pytest.mark.veryslow
        ),  # 1 million samples
    ],
    "anomalousInvalidParameters": [
        pytest.param(
            None, None, None, None, ValidationError, id="anomaly_null_input_params"
        ),
        pytest.param(0.0, 0.0, 0.0, 0.0, ValidationError, id="all_zeroes"),
        pytest.param(10.0, -3.0, 2.0, 4, ValidationError, id="all_zeroes"),
        pytest.param(-10.0, 3.0, 2.0, 4, ValidationError, id="all_zeroes"),
        pytest.param(10.0, 3.0, -2.0, 4, ValidationError, id="all_zeroes"),
        pytest.param(10.0, 3.0, 2.0, -4, ValidationError, id="all_zeroes"),
    ],
}


class TestGenerateData:
    """Test accelerometer.generate_data() and the validity of the data produced"""

    @pytest.mark.parametrize(
        "invalid_freq, invalid_time, expected_exception", TEST_CASES["invalidInputs"]
    )
    def test_invalid_inputs(
        self,
        invalid_freq: int | float,
        invalid_time: int | float,
        expected_exception: type[ValidationError],
    ):
        """Test when invalid frequency or total time provided. Assume default parameters used"""
        with pytest.raises(expected_exception) as exc_info:
            _ = generate_data(frequency=invalid_freq, total_time=invalid_time)
        assert exc_info.type is ValidationError

    @pytest.mark.parametrize(
        "valid_freq, valid_time", TEST_CASES["validInputsEmptyResult"]
    )
    def test_valid_inputs_empty_result(
        self,
        valid_freq: int | float,
        valid_time: int | float,
    ):
        """Test when valid frequency or total time provided that result in empty result. Assume default parameters used"""
        result_df = generate_data(frequency=valid_freq, total_time=valid_time)
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty
        assert len(result_df) == 0

    def test_output_structure_and_types(self):
        """Test the output DataFrame structure, columns, and basic types given default parameters."""
        frequency = 10
        time = 2
        expected_samples = frequency * time

        result_df = generate_data(frequency=frequency, total_time=time)

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == expected_samples
        assert list(result_df.columns) == [
            "timestamp",
            "sensor_id",
            "accel_x",
            "accel_y",
            "accel_z",
        ]
        assert pd.api.types.is_datetime64_any_dtype(result_df["timestamp"])
        assert result_df["timestamp"].dt.tz is not None
        assert pd.api.types.is_string_dtype(
            result_df["sensor_id"]
        ) or pd.api.types.is_object_dtype(result_df["sensor_id"])
        assert pd.api.types.is_float_dtype(result_df["accel_x"])
        assert pd.api.types.is_float_dtype(result_df["accel_y"])
        assert pd.api.types.is_float_dtype(result_df["accel_z"])
        assert isinstance(result_df.iloc[0]["timestamp"], pd.Timestamp)
        assert isinstance(result_df.iloc[0]["sensor_id"], UUID)
        assert isinstance(result_df.iloc[0]["accel_x"], (float, np.floating))

    def test_zero_noise_deterministic_output(self):
        """Test if the data produced when 0 noise is specified is predictable and constant"""
        b_roll: np.float64 = np.deg2rad(3.0)
        b_pitch: np.float64 = np.deg2rad(1.0)
        frequency = 100
        time = 1
        expected_samples = frequency * time

        params = GenerateDataParams(
            speed=0.0,
            amplitude_bounce=0.0,
            amplitude_pitch=0.0,
            amplitude_roll=0.0,
            amplitude_sway=0.0,
            noise_std_dev=0.0,
            base_roll=b_roll,
            base_pitch=b_pitch,
        )

        result_df: pd.DataFrame = generate_data(
            frequency=frequency, total_time=time, generate_data_params=params
        )
        assert len(result_df) == expected_samples

        # Calculate expected constant vector
        base_euler = [b_roll, b_pitch, 0.0]
        R_static = R.from_euler("xyz", base_euler, degrees=False).as_matrix()
        R_static = R_static.T
        gravity_vector = np.array([0, 0, -params.gravity])
        expected_vector: npt.NDArray[np.float64] = R_static @ (-gravity_vector)
        expected_x: float = expected_vector[0]
        expected_y: float = expected_vector[1]
        expected_z: float = expected_vector[2]

        assert np.allclose(result_df["accel_x"], expected_x, atol=1e-7)
        assert np.allclose(result_df["accel_y"], expected_y, atol=1e-7)
        assert np.allclose(result_df["accel_z"], expected_z, atol=1e-7)

    def test_data_distribution_with_noise(self):
        """Test if acceleration data follows desired normal distribution. Minimize variables to test base case"""
        test_noise_std_dev = 0.1
        b_roll: np.float64 = np.deg2rad(2.0)
        b_pitch: np.float64 = np.deg2rad(1.0)
        frequency = 100
        time = 20
        expected_samples = frequency * time
        params = GenerateDataParams(
            speed=0.0,
            amplitude_bounce=0.0,
            amplitude_pitch=0.0,
            amplitude_roll=0.0,
            amplitude_sway=0.0,
            base_roll=b_roll,
            base_pitch=b_pitch,
            noise_std_dev=test_noise_std_dev,
        )
        result_df = generate_data(
            frequency=frequency,
            total_time=time,
            start_time=datetime.now(),
            generate_data_params=params,
        )

        assert len(result_df) == expected_samples

        # Calculate expected mean vector
        base_euler = [b_roll, b_pitch, 0.0]
        R_static = R.from_euler("xyz", base_euler, degrees=False).as_matrix()
        R_world_to_body = R_static.T
        g_vector = np.array([0, 0, -params.gravity])
        expected_vector = R_world_to_body @ (-g_vector)
        expected_x, expected_y, expected_z = expected_vector

        # Check Mean (use higher tolerance for random sample mean)
        mean_tolerance = (
            test_noise_std_dev / np.sqrt(expected_samples) * 5
        )  # e.g., 5 sigma error
        assert np.mean(result_df["accel_x"]) == pytest.approx(
            expected_x, abs=mean_tolerance
        )
        assert np.mean(result_df["accel_y"]) == pytest.approx(
            expected_y, abs=mean_tolerance
        )
        assert np.mean(result_df["accel_z"]) == pytest.approx(
            expected_z, abs=mean_tolerance
        )

        # Check Standard Deviation (use relative tolerance)
        std_rel_tolerance = 0.20  # Allow 20% deviation for sample std dev
        assert np.std(result_df["accel_x"]) == pytest.approx(
            test_noise_std_dev, rel=std_rel_tolerance
        )
        assert np.std(result_df["accel_y"]) == pytest.approx(
            test_noise_std_dev, rel=std_rel_tolerance
        )
        assert np.std(result_df["accel_z"]) == pytest.approx(
            test_noise_std_dev, rel=std_rel_tolerance
        )

    def test_data_timestamps(self):
        """Test if the timestamps produce are distributed correctly per frequency and total time"""
        start_ts = datetime.now()
        frequency = 50
        time = 5
        expected_samples = frequency * time
        result_df: pd.DataFrame = generate_data(
            frequency=frequency, total_time=time, start_time=start_ts
        )

        assert len(result_df) == expected_samples

        expected_delta = 1.0 / frequency

        deltas_seconds = (
            np.diff(result_df["timestamp"]).astype("timedelta64[ns]").astype(np.float64)
            / 1e9
        )
        assert np.allclose(deltas_seconds, expected_delta, atol=1e-6)

        total_duration_sec = (
            result_df["timestamp"].iloc[-1] - result_df["timestamp"].iloc[0]
        ).total_seconds()
        expected_total_duration = (expected_samples - 1) * expected_delta
        assert total_duration_sec == pytest.approx(expected_total_duration)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "frequency, total_time, expected_samples", TEST_CASES["performanceTestCases"]
    )
    def test_performance(self, frequency: int, total_time: int, expected_samples: int):
        """Check if generating a largeg dataset completes within a reasonable amount of time."""
        # On M1Max MBP with 64 GB memory, got about ~0.00085 s/1k records
        # Will set threshold to 1.5ms / 1k records to account for less powerful containers running the code
        max_seconds_per_1k_records = 0.0015

        params = GenerateDataParams(noise_std_dev=0.01)

        max_duration = (expected_samples / 1000.0) * max_seconds_per_1k_records
        max_duration += 0.1

        start_time_exec = time.perf_counter()
        result_df = generate_data(
            frequency=frequency, total_time=total_time, generate_data_params=params
        )
        end_time_exec = time.perf_counter()
        duration = end_time_exec - start_time_exec

        assert len(result_df) == expected_samples, "DataFrame size mismatch"

        print(
            f"\nPerformance Test ({expected_samples} samples): Duration={duration:.4f}s, Max Expected={max_duration:.4f}s"
        )
        assert duration < max_duration, (
            f"Data generation for {expected_samples} samples took {duration:.4f}s, "
            f"exceeding proportional threshold of {max_duration:.4f}s "
            f"(based on {max_seconds_per_1k_records * 1000:.3f} ms/1k records)"
        )


class TestGenerateAnomalousData:
    """Test accelerometer.generate_data() and the validity of the anomalous data produced"""

    @pytest.mark.parametrize(
        "z_amp_modifier, time_drift_offset, step_time_delay, step_frequency, expected_exception",
        TEST_CASES["anomalousInvalidParameters"],
    )
    def test_invalid_inputs(
        self,
        z_amp_modifier: float,
        time_drift_offset: float,
        step_time_delay: float,
        step_frequency: int,
        expected_exception: type[ValidationError],
    ):
        """Test when invalid frequency or total time provided. Assume default parameters used"""
        frequency = 50
        time = 5
        with pytest.raises(expected_exception) as exc_info:
            anomalous_params = AnomalousDataModifierParams(
                z_amp_modifier=z_amp_modifier,
                time_drift_offset=time_drift_offset,
                step_time_delay=step_time_delay,
                step_frequency=step_frequency,
            )
            _ = generate_data(
                frequency=frequency,
                total_time=time,
                generate_data_params=GenerateDataParams(),
                anomalous_data_params=anomalous_params,
            )
        assert exc_info.type is ValidationError

    def test_z_limp_modifier_quality(self):
        """Test z_amp_modifier produces desired anomalous data assuming all others inputs constant"""
        frequency = 50
        time = 20

        generate_data_params = GenerateDataParams(noise_std_dev=0)
        anomalous_params = AnomalousDataModifierParams(
            z_amp_modifier=0.4, step_frequency=4
        )

        normal_data = generate_data(
            frequency=frequency,
            total_time=time,
            generate_data_params=generate_data_params,
        )
        anomalous_data = generate_data(
            frequency=frequency,
            total_time=time,
            generate_data_params=generate_data_params,
            anomalous_data_params=anomalous_params,
        )

        # Recreate limp_step_mask_vector
        num_samples = int(frequency * time)
        time_vector: npt.NDArray[np.float64] = np.linspace(
            0.0, float(time), num_samples, endpoint=False, dtype=np.float64
        )
        step_indices = np.floor(time_vector * generate_data_params.gait_frequency)
        limp_step_mask_vector: npt.NDArray[np.bool] = (
            step_indices % anomalous_params.step_frequency
        ) == (anomalous_params.step_frequency - 1)

        # Assess whether the right samples corresponding with the "limp" step are affected
        limp_samples = anomalous_data.loc[limp_step_mask_vector, "accel_z"]
        normal_samples = anomalous_data.loc[~limp_step_mask_vector, "accel_z"]
        assert len(limp_samples) > 0
        assert len(normal_samples) > len(limp_samples)
        assert limp_samples.std() < normal_samples.std()

        # Assess data distribution is properly affected
        normal_z_std = normal_data["accel_z"].std()
        anomalous_z_std = anomalous_data["accel_z"].std()
        assert anomalous_z_std < normal_z_std
        assert normal_data["accel_x"].std() == pytest.approx(
            anomalous_data["accel_x"].std(), rel=0.01
        )
        assert normal_data["accel_y"].std() == pytest.approx(
            anomalous_data["accel_y"].std(), rel=0.01
        )

    def test_time_offset_modifier_quality(self):
        """Test time_drift_offset produces desired anomalous data assuming all others inputs constant"""
        frequency = 50
        time = 20
        pass

    def test_step_delay_modifier_quality(self):
        """Test step_time_delay produces desired anomalous data assuming all others inputs constant"""
        frequency = 50
        time = 20
        pass

    def test_step_frequency_modifier_quality(self):
        """Test ste_frequency produces the desired anomalous data frequency assuming all other inputs are constant"""
        frequency = 50
        time = 20
        pass
