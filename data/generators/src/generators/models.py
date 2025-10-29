from typing import TypedDict
from datetime import datetime
from pydantic import UUID4, PositiveFloat, PositiveInt
from pydantic.dataclasses import dataclass
import numpy as np


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

    # Motion Parameters
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


@dataclass
class AnomalousDataModifierParams:
    """Data model for parameters to add anomalous data to the generated accelerometer data"""

    # Simulate a leg having a weaker push off the ground with each step
    z_amp_modifier: PositiveFloat | None = None

    # Simulate sensor error and time drift
    time_drift_offset: PositiveFloat | None = None

    # Simulate a delay caused by slower motion of a leg with each step
    step_time_delay: PositiveFloat | None = None

    # On which step the anomalous modifier(s) should apply. Default every 4th step
    step_frequency: PositiveInt = 4
    # TODO: Possibly more later
