"""
Module for generating mock accelerometer data
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4


import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
from pydantic import UUID4, BaseModel, PositiveInt
from pydantic.dataclasses import dataclass


class AccelerometerData(BaseModel):
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

    # Base Orientation - Adjust oscillation starting points
    base_pitch: float = 0.0  # radians
    base_roll: float = 0.03  # radians

    # Phase Shifts - Adjust oscillation timings
    phase_sway: float = 0.0  # radians
    phase_bounce: float = np.pi / 2  # radians
    phase_roll: float = np.pi  # radians
    phase_pitch: float = 0.0  # radians

    # Sensor Noise
    noise_std_dev: float = 0.05  # m/s^2

    # Physics
    gravity: float = 9.81  # m/s^2


# TODO: Future considerations: parallel processing, new fxn using generators to create a stream -- yield one record at a time or in specified batch sizes
def generate_data(
    frequency: PositiveInt,
    total_time: PositiveInt,
    params: GenerateDataParams | None = None,
) -> list[AccelerometerData]:
    """Generates simulated accelerometer data
    @frequency: how many data records per second to produce
    @total_time: the total time interval in seconds for which to produce data
    @as_json: flag to return the response as json

    A few assumptions:
    - Primary goal is to simulate "real enough" accelerometer data without having to simulate the entire robot
    - This is for a quadriped robot walking on horizontal, flat plane at a constant pace
    - Given this, we assume the robot's walking pattern follows a Simple Harmonic Motion pattern and apply the SHM formula

    """
    if not params:
        params = GenerateDataParams()

    # Determine total number of samples/records to generate
    num_samples: int = int(frequency * total_time)
    if num_samples <= 0:
        return []

    # Calculate constants
    gravity_vector = np.array([0, 0, -params.gravity])
    sensor_id = uuid4()
    start_ts = datetime.now(timezone.utc)

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
    # a_prop = R_world_to_body @ (a_linear_world - g_vector)
    accel_diff_world = a_linear_world - gravity_vector
    a_prop: npt.NDArray[np.float64] = np.einsum(
        "nij, nj->ni", R_world_to_body, accel_diff_world
    )

    # Add sensor noise
    noise = np.random.normal(0, params.noise_std_dev, size=(num_samples, 3))

    a_final = a_prop + noise

    # Format output
    time_deltas: list[timedelta] = []
    for sec in time_vector:
        time_deltas.append(timedelta(seconds=sec))
    # time_deltas = [timedelta(seconds=sec) for sec in time_vector]
    output: list[AccelerometerData] = [
        AccelerometerData(
            timestamp=start_ts + time_deltas[i],
            sensor_id=sensor_id,
            accel_x=a_final[i, 0],
            accel_y=a_final[i, 1],
            accel_z=a_final[i, 2],
        )
        for i in range(num_samples)
    ]

    return output
