"""
Module for generating mock accelerometer data
"""

from datetime import datetime, timezone
from typing import TypedDict
from uuid import uuid4


import numpy as np
import pandas as pd
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
from pydantic import UUID4, PositiveFloat, PositiveInt, validate_call
from pydantic.dataclasses import dataclass


class AccelerometerData(TypedDict):
    """Data model for accelerometer time series data"""

    timestamp: datetime
    sensor_id: UUID4
    accel_x: float
    accel_y: float
    accel_z: float


@dataclass
class GenerateDataParams:
    """Data model for accelerometer-specific arguments for generating data"""

    # Motion Parameters -
    gait_frequency: float = 2.0  # steps/sec
    speed: float = 1.0  # m/s
    base_height: float = 0.5  # meters

    # Oscillation Amplitudes
    amplitude_sway: float = 0.05  # meters
    amplitude_bounce: float = 0.02  # meters
    amplitude_roll: float = 0.05  # radians
    amplitude_pitch: float = 0.05  # radians

    # Base Orientation - Adjusts oscillation starting points
    base_pitch: float = 0.0  # radians
    base_roll: float = 0.03  # radians

    # Phase Shifts - Adjusts oscillation timings
    phase_sway: float = 0.0  # radians
    phase_bounce: float = np.pi / 2  # radians
    phase_roll: float = np.pi  # radians
    phase_pitch: float = 0.0  # radians

    # Sensor Noise
    noise_std_dev: float = 0.05  # m/s^2

    # Physics
    gravity: float = 9.81  # m/s^2


# TODO: Future considerations: parallel processing, new fxn using generators to create a stream -- yield one record at a time or in specified batch sizes
# TODO: Consider timezone-aware timestamps. For real-world scenario of data coming from robots across multiple timezones
@validate_call
def generate_data(
    frequency: PositiveInt | PositiveFloat,
    total_time: PositiveInt | PositiveFloat,
    start_time: datetime | None = None,
    params: GenerateDataParams | None = None,
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
    if not params:
        params = GenerateDataParams()

    # Determine total number of samples/records to generate
    num_samples: int = int(frequency * total_time)
    if num_samples <= 0:
        return pd.DataFrame()

    # Calculate constants
    gravity_vector = np.array([0, 0, -params.gravity])
    sensor_id = uuid4()
    start_ts = start_time if start_time else datetime.now(timezone.utc)

    # Generate time vector
    time_vector: npt.NDArray[np.float64] = np.linspace(
        0.0, float(total_time), num_samples, endpoint=False, dtype=np.float64
    )

    # Calculate robot body orientation given by Euler angles (radians) over time assuming SHM
    # Convert gait_frequency to angular frequency (ω) of SHM
    omega_gait = 2 * np.pi * params.gait_frequency

    roll = params.base_roll + params.amplitude_roll * np.sin(
        omega_gait * time_vector + params.phase_roll
    )
    pitch = params.base_pitch + params.amplitude_pitch * np.sin(
        omega_gait * time_vector + params.phase_pitch
    )
    yaw = np.zeros_like(
        time_vector
    )  # Assume robot is walking in a straight line with no turning

    # Calculate Rotation Matrices (Robot body frame TO World frame)
    euler_orientations = np.stack([roll, pitch, yaw], axis=-1)
    R_body_world = R.from_euler("xyz", euler_orientations, degrees=False).as_matrix()
    R_world_to_body = R_body_world.transpose(
        (0, 2, 1)
    )  # Transpose to be able to go from World frame TO body frame

    # How often the body sways and bounces. Often relative to gait
    omega_sway = 2 * np.pi * (params.gait_frequency / 2)
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
        -params.amplitude_sway
        * (omega_sway**2)
        * np.sin(omega_sway * time_vector + params.phase_sway)
    )
    accel_z = (
        -params.amplitude_bounce
        * (omega_bounce**2)
        * np.sin(omega_bounce * time_vector + params.phase_bounce)
    )

    a_linear_world = np.stack([accel_x, accel_y, accel_z], axis=-1)

    # Calculate proper acceleration (in body frame)
    # For reference: a_prop = R_world_to_body @ (a_linear_world - gravity_vector)
    # Using Einstein Summation method instead
    accel_diff_world = a_linear_world - gravity_vector
    a_prop: npt.NDArray[np.float64] = np.einsum(
        "nij, nj->ni", R_world_to_body, accel_diff_world
    )

    # Add sensor noise
    noise = np.random.normal(0, params.noise_std_dev, size=(num_samples, 3))

    # Calculate final acceleration matrix
    a_final = a_prop + noise

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
