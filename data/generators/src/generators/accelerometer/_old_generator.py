"""
Module for generating mock accelerometer data
"""

from datetime import datetime, timezone
import logging
from uuid import uuid4

import numpy as np
import pandas as pd
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
from pydantic import PositiveFloat, PositiveInt, validate_call

from .models import AnomalousDataModifierParams, GenerateDataParams

logger = logging.getLogger(__name__)


def apply_time_anomalies(
    time_vector: npt.NDArray[np.float64],
    gait_frequency: float,
    anomalous_data_params: AnomalousDataModifierParams | None,
) -> npt.NDArray[np.float64]:
    """Function to apply time-related anomalous modifiers to the generated data"""
    if not anomalous_data_params:
        return time_vector

    logger.info("Applying time-based anomalies...")
    modified_time_vector: npt.NDArray[np.float64] = time_vector.copy()

    # Apply cumulative time drift
    if (
        anomalous_data_params.time_drift_offset
        and anomalous_data_params.time_drift_offset > 0
    ):
        drift = (
            np.arange(len(modified_time_vector))
            * anomalous_data_params.time_drift_offset
        )
        modified_time_vector += drift
        logger.debug("Applied time drift.")

    # Apply intermittent step delay
    if (
        anomalous_data_params.step_time_delay
        and anomalous_data_params.step_time_delay > 0
    ):
        step_indices = np.floor(time_vector * gait_frequency)
        delayed_step_mask_vector: npt.NDArray[np.bool] = (
            step_indices % anomalous_data_params.step_frequency
        ) == (anomalous_data_params.step_frequency - 1)
        delay_to_add: npt.NDArray[np.float64] = np.where(
            delayed_step_mask_vector, anomalous_data_params.step_time_delay, 0.0
        )
        modified_time_vector += delay_to_add
        logger.debug("Applied intermittent step delay.")

    return modified_time_vector


def apply_amplitude_anomalies(
    base_amplitude: float,
    time_vector: npt.NDArray[np.float64],
    gait_frequency: float,
    anomalous_data_params: AnomalousDataModifierParams | None,
) -> npt.NDArray[np.float64] | float:
    """
    Function to apply amplitude-related anomalous modifiers to the generated data.
    """
    # Modify Z-axis amplitude if applicable
    if not anomalous_data_params or not anomalous_data_params.z_amp_modifier:
        # return scalar if no modification to apply
        return base_amplitude

    logger.info("Applying amplitude-based anomalies")

    modified_amplitude = base_amplitude * anomalous_data_params.z_amp_modifier
    step_indices = np.floor(time_vector * gait_frequency)
    limp_step_mask_vector: npt.NDArray[np.bool] = (
        step_indices % anomalous_data_params.step_frequency
    ) == (anomalous_data_params.step_frequency - 1)

    return np.where(limp_step_mask_vector, modified_amplitude, base_amplitude)


# TODO: Future considerations: parallel processing, new fxn using generators to create a stream -- yield one record at a time or in specified batch sizes
# TODO: Consider timezone-aware timestamps. For real-world scenario of data coming from robots across multiple timezones
@validate_call
def generate_data(
    frequency: PositiveInt | PositiveFloat,
    total_time: PositiveInt | PositiveFloat,
    start_time: datetime | None = None,
    generate_data_params: GenerateDataParams | None = None,
    anomalous_data_params: AnomalousDataModifierParams | None = None,
) -> pd.DataFrame:
    """Generates simulated accelerometer data

    A few assumptions:
    - Primary goal is to simulate "real enough" accelerometer data without having to simulate the entire robot
    - This is for a quadriped robot walking on horizontal, flat plane at a constant pace
    - Given this, we assume the robot's walking pattern follows a Simple Harmonic Motion pattern and apply the SHM formula

    :param frequency: How many data records per second to produce
    :param total_time: The total time interval in seconds for which to produce data
    :param start_time: The desired start timestamp for the sample data. Defaults to the timestamp of when generate_data is called.
    :param params: Parameters specific to generating accelerometer data such as sway, bounce, roll, and pitch parameters.
    :returns: Returns a pandas DataFrame containing x,y,z acceleration values along with a sensor ID and timestamp
    """
    if not generate_data_params:
        generate_data_params = GenerateDataParams()

    # Determine total number of samples/records to generate
    num_samples: int = int(frequency * total_time)
    if num_samples <= 0:
        return pd.DataFrame()

    # Constants
    gravity_vector = np.array([0, 0, -generate_data_params.gravity_mps2])
    sensor_id = uuid4()
    start_ts = start_time if start_time else datetime.now(timezone.utc)

    # Generate time vector
    time_vector: npt.NDArray[np.float64] = np.linspace(
        0.0, float(total_time), num_samples, endpoint=False, dtype=np.float64
    )

    time_vector = apply_time_anomalies(
        time_vector, generate_data_params.gait_frequency_hz, anomalous_data_params
    )

    # Calculate robot body orientation given by Euler angles (radians) over time assuming SHM
    # Convert gait_frequency to angular frequency (ω) of SHM
    omega_gait = 2 * np.pi * generate_data_params.gait_frequency_hz

    roll = (
        generate_data_params.base_roll_rad
        + generate_data_params.amplitude_roll_rad
        * np.sin(omega_gait * time_vector + generate_data_params.phase_roll_rad)
    )
    pitch = (
        generate_data_params.base_pitch_rad
        + generate_data_params.amplitude_pitch_rad
        * np.sin(omega_gait * time_vector + generate_data_params.phase_pitch_rad)
    )
    yaw = np.zeros_like(
        time_vector
    )  # Assume robot is walking in a straight line with no turning

    # Calculate Rotation Matrices (Robot body frame TO World frame)
    euler_orientations = np.stack([roll, pitch, yaw], axis=-1)
    R_body_to_world = R.from_euler("xyz", euler_orientations, degrees=False).as_matrix()
    R_world_to_body = R_body_to_world.transpose(
        (0, 2, 1)
    )  # Transpose to be able to go from World frame TO body frame

    # How often the body sways and bounces. Often relative to gait
    omega_sway = 2 * np.pi * (generate_data_params.gait_frequency_hz / 2)
    omega_bounce = omega_gait

    # For reference: The formulas for calculating x,y,z position at time t
    # x_pos = params.speed * time_vector
    # y_pos = params.amplitude_sway * np.sin(omega_sway * time_vector + params.phase_sway)
    # z_pos = params.base_height + params.amplitude_bounce * np.sin(
    #     omega_bounce * time_vector + params.phase_bounce
    # )

    # Calculate linear acceleration in world frame - 2nd derivative of position at time t
    # d^2(A*sin(ω*t + p))dt^2 = -A*ω^2*sin(ω*t + p)
    accel_x = np.zeros_like(time_vector)  # speed is constant
    accel_y = (
        -generate_data_params.amplitude_sway_m
        * (omega_sway**2)
        * np.sin(omega_sway * time_vector + generate_data_params.phase_sway_rad)
    )

    amplitude_z: npt.NDArray[np.float64] | float = apply_amplitude_anomalies(
        generate_data_params.amplitude_bounce_m,
        time_vector,
        generate_data_params.gait_frequency_hz,
        anomalous_data_params,
    )
    accel_z = (
        -amplitude_z
        * (omega_bounce**2)
        * np.sin(omega_bounce * time_vector + generate_data_params.phase_bounce_rad)
    )

    a_linear_world = np.stack([accel_x, accel_y, accel_z], axis=-1)

    # Calculate proper acceleration (in body frame)
    # For reference: a_prop = R_world_to_body @ (a_linear_world - gravity_vector)
    # Using Einstein Summation method instead
    accel_diff_world = a_linear_world - gravity_vector
    a_proper: npt.NDArray[np.float64] = np.einsum(
        "nij, nj->ni", R_world_to_body, accel_diff_world
    )

    # Add sensor noise
    noise = np.random.normal(
        0, generate_data_params.noise_std_dev, size=(num_samples, 3)
    )

    # Calculate final acceleration matrix
    a_final = a_proper + noise

    # Generate timestamps
    start_ts_np = np.datetime64(start_ts, "ns")
    time_deltas = (time_vector * 1e9).astype("timedelta64[ns]")
    timestamps: npt.NDArray[np.datetime64] = start_ts_np + time_deltas
    # Format output
    sensor_ids = np.full(num_samples, sensor_id, dtype=object)
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "sensor_id": sensor_ids,
            "accel_x": a_final[:, 0],
            "accel_y": a_final[:, 1],
            "accel_z": a_final[:, 2],
        }
    )
    # Numpy datetime64 objects are not timezone-aware So we must re-add the timezone in formation
    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    return df
