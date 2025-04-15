from uuid import UUID
from generators.accelerometer import (
    AccelerometerData,
    GenerateDataParams,
    generate_data,
)
from datetime import datetime

import pytest
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
from pydantic import ValidationError


TEST_CASES = {
    "invalidInputs": [
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
        pytest.param(1.0, 0.5, [], id="floats_resulting_zero_samples_1"),
        pytest.param(1.99, 0.5, [], id="float_resulting_zero_samples_2"),
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
        "valid_freq, valid_time, expected_output", TEST_CASES["validInputsEmptyResult"]
    )
    def test_valid_inputs_empty_result(
        self,
        valid_freq: int | float,
        valid_time: int | float,
        expected_output: list[AccelerometerData],
    ):
        """Test when valid frequency or total time provided that result in empty result. Assume default parameters used"""
        result = generate_data(frequency=valid_freq, total_time=valid_time)
        assert expected_output == result

    def test_default_parameters(self):
        """Test if the default parameters produce valid data"""
        frequency = 50
        time = 2
        expected_samples = frequency * time
        result = generate_data(frequency=frequency, total_time=time)

        assert isinstance(result, list)
        assert len(result) == expected_samples

        if expected_samples > 0:
            for data in [result[0], result[-1]]:
                assert isinstance(data, AccelerometerData)
                assert isinstance(data.timestamp, datetime)
                assert isinstance(data.sensor_id, UUID)
                assert isinstance(data.accel_x, float)

    def test_zero_noise(self):
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

        result: list[AccelerometerData] = generate_data(
            frequency=frequency, total_time=time, params=params
        )
        assert len(result) == expected_samples

        # Calculate expected constant vector
        base_euler = [b_roll, b_pitch, 0.0]
        R_static = R.from_euler("xyz", base_euler, degrees=False).as_matrix()
        R_static = R_static.T
        g_vector = np.array([0, 0, -params.gravity])
        expected_vector: npt.NDArray[np.float64] = R_static @ (-g_vector)
        expected_x: float = expected_vector[0]
        expected_y: float = expected_vector[1]
        expected_z: float = expected_vector[2]

        for data in result:
            assert data.accel_x == pytest.approx(expected_x)
            assert data.accel_y == pytest.approx(expected_y)
            assert data.accel_z == pytest.approx(expected_z)

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
        result = generate_data(
            frequency=frequency,
            total_time=time,
            start_time=datetime.now(),
            params=params,
        )

        assert len(result) == expected_samples

        # Calculate expeected mean vector
        base_euler = [b_roll, b_pitch, 0.0]
        R_static = R.from_euler("xyz", base_euler, degrees=False).as_matrix()
        R_static = R_static.T
        g_vector = np.array([0, 0, -params.gravity])
        expected_vector = R_static @ (-g_vector)
        expected_x: float = expected_vector[0]
        expected_y: float = expected_vector[1]
        expected_z: float = expected_vector[2]

        all_ax = np.array([d.accel_x for d in result], dtype=np.float64)
        all_ay = np.array([d.accel_y for d in result], dtype=np.float64)
        all_az = np.array([d.accel_z for d in result], dtype=np.float64)

        assert np.mean(all_ax) == pytest.approx(
            expected_x, abs=test_noise_std_dev / np.sqrt(expected_samples) * 4
        )  # Allow some tolerance based on sample size
        assert np.mean(all_ay) == pytest.approx(
            expected_y, abs=test_noise_std_dev / np.sqrt(expected_samples) * 4
        )  # (Using 4*std error of mean as tolerance)
        assert np.mean(all_az) == pytest.approx(
            expected_z, abs=test_noise_std_dev / np.sqrt(expected_samples) * 4
        )

        assert np.std(all_ax) == pytest.approx(
            test_noise_std_dev, rel=0.15
        )  # Allow relative tolerance (e.g., 15%) for std dev estimate
        assert np.std(all_ay) == pytest.approx(test_noise_std_dev, rel=0.15)
        assert np.std(all_az) == pytest.approx(test_noise_std_dev, rel=0.15)

        all_ax = np.array([d.accel_x for d in result], dtype=np.float64)
        all_ay = np.array([d.accel_y for d in result], dtype=np.float64)
        all_az = np.array([d.accel_z for d in result], dtype=np.float64)

    def test_data_timestamps(self):
        """Test if the timestamps produce are distributed correctly per frequency and total time"""
        start_ts = datetime.now()
        frequency = 50
        time = 5
        expected_samples = frequency * time
        result: list[AccelerometerData] = generate_data(
            frequency=frequency, total_time=time, start_time=start_ts
        )

        assert len(result) == expected_samples

        expected_delta = 1.0 / frequency

        # Check if time deltas between samples is as expected
        deltas_seconds = np.diff([data.timestamp.timestamp() for data in result])
        assert np.allclose(deltas_seconds, expected_delta, atol=1e-6)

        # Check total duration is correct
        total_duration = (result[-1].timestamp - result[0].timestamp).total_seconds()
        expected_total_duration = (len(result) - 1) * expected_delta
        assert total_duration == pytest.approx(expected_total_duration)
