import math
from typing import TypedDict
from datetime import datetime
from dataclasses import dataclass
from uuid import UUID

import numpy as np


class AccelerometerData(TypedDict):
    """Data model for accelerometer time series data"""

    timestamp: datetime
    sensor_id: UUID
    accel_x: float
    accel_y: float
    accel_z: float


def _norm_phase(radians: float) -> float:
    """Normalize angle to [0, 2pi]"""
    two_pi = math.tau
    r = radians % two_pi
    return r if r >= 0 else r + two_pi


@dataclass(slots=True, frozen=True)
class GenerateDataParams:
    """Data model for accelerometer-specific arguments for generating data"""

    # Motion Parameters
    gait_frequency_hz: float = 2.0  # steps/sec
    # speed_mps: float = 4.0  # meters/sec. Unused. May need for future improvements to accelerometer model

    # Oscillation Amplitudes
    amplitude_sway_m: float = 0.05  # meters
    amplitude_bounce_m: float = 0.02  # meters
    amplitude_roll_rad: float = 0.05  # radians
    amplitude_pitch_rad: float = 0.05  # radians

    # Base Orientation - Adjusts oscillation starting points
    base_pitch_rad: float = 0.0  # radians
    base_roll_rad: float = 0.03  # radians
    # base_height_m: float = (
    #     0.5  # meters. Unused. May need for future improvements of accelerometer model
    # )

    # Phase Shifts - Adjusts oscillation timings
    phase_sway_rad: float = 0.0  # radians
    phase_bounce_rad: float = np.pi / 2  # radians
    phase_roll_rad: float = np.pi  # radians
    phase_pitch_rad: float = 0.0  # radians

    # Sensor Noise
    noise_std_dev: float = 0.05

    # Physics
    gravity_mps2: float = 9.81  # m/s^2

    def __post_init__(self):
        if self.gait_frequency_hz <= 0:
            raise ValueError("gait_frequency_hz must be > 0")
        if self.amplitude_sway_m < 0:
            raise ValueError("amplitude_sway_m must be >= 0")
        if self.amplitude_bounce_m < 0:
            raise ValueError("amplitude_bounce_m must be >= 0")
        if self.amplitude_roll_rad < 0:
            raise ValueError("amplitude_roll_rad must be >= 0")
        if self.amplitude_pitch_rad < 0:
            raise ValueError("amplitude_pitch_rad must be >= 0")
        if self.noise_std_dev < 0:
            raise ValueError("noise_std_dev must be >= 0")
        if self.gravity_mps2 <= 0:
            raise ValueError("gravity_mps2 must be > 0")

        object.__setattr__(self, "phase_sway_rad", _norm_phase(self.phase_sway_rad))
        object.__setattr__(self, "phase_bounce_rad", _norm_phase(self.phase_bounce_rad))
        object.__setattr__(self, "phase_roll_rad", _norm_phase(self.phase_roll_rad))
        object.__setattr__(self, "phase_pitch_rad", _norm_phase(self.phase_pitch_rad))


@dataclass(slots=True, frozen=True)
class AnomalousDataModifierParams:
    """Data model for parameters to add anomalous data to the generated accelerometer data"""

    # Simulate a leg having a weaker push off the ground with each step
    z_amp_modifier: float | None = None

    # Simulate sensor error and time drift
    time_drift_offset: float | None = None

    # Simulate a delay caused by slower motion of a leg with each step
    step_time_delay: float | None = None

    # On which step the anomalous modifier(s) should apply. Default every 4th step
    step_frequency: int = 4

    def __post_init__(self):
        if self.step_frequency <= 0:
            raise ValueError("step_frequency must be > 0")
        if self.step_time_delay is not None and self.step_time_delay < 0:
            raise ValueError("step_time_delay must be >= 0")
        if self.time_drift_offset is not None and self.time_drift_offset < 0:
            raise ValueError("time_drift_offset must be >= 0")
        if self.z_amp_modifier is not None and self.z_amp_modifier <= 0:
            raise ValueError("z_amp_modifier must be > 0 when set")
