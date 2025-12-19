"""
Module for generating simulated accelerometer data in a stream.
Defaults to generating data in 'real-time', but can generate batches of real-time data
as fast as python can do it for simplicity's sake.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timedelta, timezone
from itertools import islice
import math
import time
from uuid import UUID

import numpy as np
import pandas as pd
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from generators.accelerometer.models import (
    AnomalousDataModifierParams,
    GenerateDataParams,
    AnomalyState,
    StreamState,
    StreamStartParameters,
    AccelerometerDataPoint,
)


class AccelerometerGenerator:
    """
    Main class for generating accelerometer data.
    """

    def __init__(
        self,
        sensor_id: UUID,
        generate_data_params: GenerateDataParams | None = None,
        anomaly_data_params: AnomalousDataModifierParams | None = None,
        anomaly_state: AnomalyState | None = None,
        stream_state: StreamState | None = None,
    ):
        self.sensor_id: UUID = sensor_id
        self.sensor_type_id: str = "accelerometer"  # Soon(TM)
        self.anomaly_state: AnomalyState = (
            anomaly_state if anomaly_state else AnomalyState()
        )
        self.generate_data_params: GenerateDataParams = (
            generate_data_params if generate_data_params else GenerateDataParams()
        )
        self.anomaly_data_params: AnomalousDataModifierParams | None = (
            anomaly_data_params
        )

        self.stream_state: StreamState | None = stream_state
        self._stop: bool = False

    def start(self):
        self._stop = False

    def stop(self):
        self._stop = True

    def _effective_time_seconds(
        self,
        elapsed_time: float,  # seconds
    ) -> float:
        """
        Derive effective time from the monotonic elapsed time.
        If any time-related anomalous parameters have been specified, apply them.

        :param elapsed_time: The current amount of monotonic time that has passed so far.
        :param sample_idx: The current index of the sample data being produced. **NOTE**: This value is a function
        of the sample rate frequency and time and should not be confused with the step_idx, a function of gait_frequency.
        :returns: A float value representing the total effective time that has passed accounting for anomalies.
        """
        if not self.anomaly_data_params:
            return elapsed_time

        elapsed_time_sec = elapsed_time

        # Update cumulative time drift
        if (
            self.anomaly_data_params.time_drift_offset
            and self.anomaly_data_params.time_drift_offset > 0
            and self.stream_state
        ):
            self.anomaly_state.cumulative_time_drift = (
                self.stream_state.sample_index
                * self.anomaly_data_params.time_drift_offset
            )

        # Update cumulative step delay
        if (
            self.anomaly_data_params.step_time_delay
            and self.anomaly_data_params.step_time_delay > 0
            and self.anomaly_data_params.step_frequency
            and self.anomaly_data_params.step_frequency > 0
        ):
            step_idx = int(
                math.floor(
                    elapsed_time_sec * self.generate_data_params.gait_frequency_hz
                )
            )
            if self.anomaly_state.last_step_idx_seen is None:
                self.anomaly_state.last_step_idx_seen = step_idx

            if step_idx != self.anomaly_state.last_step_idx_seen:
                prev = self.anomaly_state.last_step_idx_seen
                if (prev % self.anomaly_data_params.step_frequency) == (
                    self.anomaly_data_params.step_frequency - 1
                ):
                    self.anomaly_state.cumulative_step_delay += (
                        self.anomaly_data_params.step_time_delay
                    )
                self.anomaly_state.last_step_idx_seen = step_idx

        return (
            elapsed_time_sec
            + self.anomaly_state.cumulative_time_drift
            + self.anomaly_state.cumulative_step_delay
        )

    def _z_amplitude_at_time_t(
        self,
        t_eff: float,
    ) -> float:
        """
        Calculate the amplitude of the Z-Axis accelerometer data, applying anomaly modifiers if specified.

        :param t_eff: The effective time as perceived by the robot. Differs from real-world time if
        time-related anomalies are specified and applied.
        :returns: The Z-Axis amplitude at the effective time t_eff.
        """
        if not self.anomaly_data_params or not self.anomaly_data_params.z_amp_modifier:
            return self.generate_data_params.amplitude_bounce_m

        step_idx = int(math.floor(t_eff * self.generate_data_params.gait_frequency_hz))
        if (step_idx % self.anomaly_data_params.step_frequency) == (
            self.anomaly_data_params.step_frequency - 1
        ):
            return (
                self.generate_data_params.amplitude_bounce_m
                * self.anomaly_data_params.z_amp_modifier
            )
        return self.generate_data_params.amplitude_bounce_m

    def _calculate_data_point(
        self,
        t_eff: float,
        omega_gait: float,
        omega_sway: float,
        omega_bounce: float,
        gravity_world: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """
        Core logic for calculating the the simulated accelerometer readings at the effective time t_eff.
        Starts by calculating roll, pitch, yaw in world frame, then accleration along x, y, and z axes
        in the world frame, before finally converting acceleration to the body frame and adding
        simulated noise.

        :param t_eff: The effective time as percieved by the robot. Differs from real-world time if
        time-related anomalies are specified and applied.
        :param rng: NumPy Random Number Generator. Serves as consistent RNG across sample generation calls.
        :returns: dictionary containing the acceleration values at time t_eff along with the timestamp
        """
        # Euler angles (roll, pitch, yaw=0) with Simple Harmonic Motion (SHM) formula
        roll = (
            self.generate_data_params.base_roll_rad
            + self.generate_data_params.amplitude_roll_rad
            * math.sin(omega_gait * t_eff + self.generate_data_params.phase_roll_rad)
        )
        pitch = (
            self.generate_data_params.base_pitch_rad
            + self.generate_data_params.amplitude_pitch_rad
            * math.sin(omega_gait * t_eff + self.generate_data_params.phase_pitch_rad)
        )
        yaw = 0.0

        # Rotation matrices
        R_bw = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()
        R_wb = R_bw.T

        # Amplitude of Z-axis sinusoidal wave accounting for anomalous_params.z_amp_modifier
        amp_z = self._z_amplitude_at_time_t(t_eff)

        # For reference: The SHM formulas for calculating x,y,z position at time t
        # x_pos = params.speed * time_vector
        # y_pos = params.amplitude_sway * np.sin(omega_sway * time_vector + params.phase_sway)
        # z_pos = params.base_height + params.amplitude_bounce * np.sin(
        #     omega_bounce * time_vector + params.phase_bounce
        # )

        # Calculate linear acceleration in world frame - 2nd derivative of position at time t
        # d^2(A*sin(ω*t + p))dt^2 = -A*ω^2*sin(ω*t + p)

        # Linear acceleration in world frame
        ax_world = 0.0
        ay_world = (
            -self.generate_data_params.amplitude_sway_m
            * (omega_sway**2)
            * math.sin(omega_sway * t_eff + self.generate_data_params.phase_sway_rad)
        )
        az_world = (
            -amp_z
            * (omega_bounce**2)
            * math.sin(
                omega_bounce * t_eff + self.generate_data_params.phase_bounce_rad
            )
        )
        a_world = np.array([ax_world, ay_world, az_world], dtype=np.float64)

        # Convert accleration from world frame to body frame
        a_proper = R_wb @ (a_world - gravity_world)

        # Add noise
        noise = rng.normal(
            loc=0.0, scale=self.generate_data_params.noise_std_dev, size=3
        )
        a_final = a_proper + noise

        return a_final

    def _prime_stream(
        self, start_time: datetime | None, data_frequency: float, rng_seed: int | None
    ) -> StreamStartParameters:
        """
        Helper function to set up required variables for data streams.

        :param start_time: Optional start time that the producer can start at.
        :param data_frequency: The rate at which to produce the data stream in records/second.
        :param rng_seed: Optional seed number for random number generator
        """
        if data_frequency <= 0:
            raise ValueError("data_frequency must be greather than 0")
        start_ts_utc = start_time or datetime.now(timezone.utc)
        start_mono = time.monotonic()
        period = 1.0 / data_frequency
        if self.stream_state is None:
            self.stream_state = StreamState(
                sensor_id=self.sensor_id,
                start_ts_utc=start_ts_utc,
                start_mono=start_mono,
                anomaly_state=AnomalyState(),
                sample_index=0,
            )
        rng = np.random.default_rng(rng_seed)
        g = self.generate_data_params.gravity_mps2
        gravity_world = np.array([0.0, 0.0, -g], dtype=np.float64)

        # Angular frequencies (ω values)
        omega_gait = 2 * np.pi * self.generate_data_params.gait_frequency_hz
        omega_sway = 2 * np.pi * (self.generate_data_params.gait_frequency_hz / 2)
        omega_bounce = omega_gait

        return {
            "start_mono": start_mono,
            "rng": rng,
            "period": period,
            "gravity_world": gravity_world,
            "omega_gait": omega_gait,
            "omega_sway": omega_sway,
            "omega_bounce": omega_bounce,
        }

    def generate_data_stream(
        self,
        *,
        data_frequency: float,
        start_time: datetime | None = None,
        rng_seed: int | None = None,
        real_time: bool = True,
    ) -> Iterator[AccelerometerDataPoint]:
        """
        Synchronous producer of a stream of acclerometer data points.

        :param data_frequency: The rate at which to produce the data stream in records/second.
        :param start_time: Optional start time that the producer can start at.
        :param rng_seed: Optional seed number for random number generator
        :param real_time: Boolean flag denoting whether to simulate real-time stream or not.
        If false then data will be produced as fast as Python can do it.
        :returns: Dictionary with structure specified by AcclerometerDataPoint class
        """

        stream_params = self._prime_stream(start_time, data_frequency, rng_seed)
        if self.stream_state is None:
            raise AttributeError("stream_state was not properly initialized.")

        next_deadline = stream_params["start_mono"]
        while not self._stop:
            if real_time:
                now = time.monotonic()
                if now < next_deadline:
                    time.sleep(next_deadline - now)
                    now = time.monotonic()
                t_real = time.monotonic() - self.stream_state.start_mono
            else:
                # In non-realtime mode, simulate proper time intervals based on sample rate
                t_real = self.stream_state.sample_index * stream_params["period"]

            t_eff = self._effective_time_seconds(
                t_real,
            )

            a_body = self._calculate_data_point(
                t_eff,
                stream_params["omega_gait"],
                stream_params["omega_sway"],
                stream_params["omega_bounce"],
                stream_params["gravity_world"],
                stream_params["rng"],
            )

            timestamp = self.stream_state.start_ts_utc + timedelta(seconds=t_eff)

            yield {
                "timestamp": timestamp,
                "sensor_id": self.stream_state.sensor_id,
                "accel_x": float(a_body[0]),
                "accel_y": float(a_body[1]),
                "accel_z": float(a_body[2]),
                "sequence": self.stream_state.sample_index,
            }

            self.stream_state.sample_index += 1
            next_deadline += stream_params["period"]

    async def async_generate_data_stream(
        self,
        *,
        data_frequency: float,
        start_time: datetime | None = None,
        rng_seed: int | None = None,
        real_time: bool = True,
    ) -> AsyncIterator[AccelerometerDataPoint]:
        """
        Synchronous producer of a stream of acclerometer data points.

        :param data_frequency: The rate at which to produce the data stream in records/second.
        :param start_time: Optional start time that the producer can start at.
        :param rng_seed: Optional seed number for random number generator
        :param real_time: Boolean flag denoting whether to simulate real-time stream or not.
        If false then data will be produced as fast as Python can do it.
        :returns: Dictionary with structure specified by AcclerometerDataPoint class
        """
        stream_params = self._prime_stream(start_time, data_frequency, rng_seed)
        if self.stream_state is None:
            raise AttributeError("stream_state was not properly initialized.")

        next_deadline = stream_params["start_mono"]
        while not self._stop:
            if real_time:
                now = time.monotonic()
                if now < next_deadline:
                    await asyncio.sleep(next_deadline - now)
                    now = time.monotonic()
                t_real = time.monotonic() - self.stream_state.start_mono
            else:
                # In non-realtime mode, simulate proper time intervals based on sample rate
                t_real = self.stream_state.sample_index * stream_params["period"]

            t_eff = self._effective_time_seconds(
                t_real,
            )

            a_body = self._calculate_data_point(
                t_eff,
                stream_params["omega_gait"],
                stream_params["omega_sway"],
                stream_params["omega_bounce"],
                stream_params["gravity_world"],
                stream_params["rng"],
            )

            timestamp = self.stream_state.start_ts_utc + timedelta(seconds=t_eff)

            yield {
                "timestamp": timestamp,
                "sensor_id": self.stream_state.sensor_id,
                "accel_x": float(a_body[0]),
                "accel_y": float(a_body[1]),
                "accel_z": float(a_body[2]),
                "sequence": self.stream_state.sample_index,
            }

            self.stream_state.sample_index += 1
            next_deadline += stream_params["period"]

    def batch_stream(
        self,
        *_,
        data_frequency: float,
        start_time: datetime | None = None,
        rng_seed: int | None = None,
        real_time: bool = True,
        batch_size: int,
    ) -> Iterator[list[AccelerometerDataPoint]]:
        """
        Helper function to bucketize stream data.

        :param data_frequency: The rate at which to produce the data stream in records/second.
        :param start_time: Optional start time that the producer can start at.
        :param rng_seed: Optional seed number for random number generator
        :param real_time: Boolean flag denoting whether to simulate real-time stream or not.
        If false then data will be produced as fast as Python can do it.
        :param batch_size: number of records per bucket
        :returns: Iterator of lists (buckets) of AccelerometerDataPoint objects
        """
        sink: list[AccelerometerDataPoint] = []
        for rec in self.generate_data_stream(
            *_,
            data_frequency=data_frequency,
            start_time=start_time,
            rng_seed=rng_seed,
            real_time=real_time,
        ):
            sink.append(rec)
            if len(sink) >= batch_size:
                yield sink
                sink = []

    async def async_batch_stream(
        self,
        *_,
        data_frequency: float,
        start_time: datetime | None = None,
        rng_seed: int | None = None,
        real_time: bool = True,
        batch_size: int,
    ) -> AsyncIterator[list[AccelerometerDataPoint]]:
        """
        Helper function to bucketize async stream data.

        :param data_frequency: The rate at which to produce the data stream in records/second.
        :param start_time: Optional start time that the producer can start at.
        :param rng_seed: Optional seed number for random number generator
        :param real_time: Boolean flag denoting whether to simulate real-time stream or not.
        If false then data will be produced as fast as Python can do it.
        :param batch_size: number of records per bucket
        :returns: Iterator of lists (buckets) of AccelerometerDataPoint objects
        """
        sink: list[AccelerometerDataPoint] = []
        async for rec in self.async_generate_data_stream(
            *_,
            data_frequency=data_frequency,
            start_time=start_time,
            rng_seed=rng_seed,
            real_time=real_time,
        ):
            sink.append(rec)
            if len(sink) >= batch_size:
                yield sink
                sink = []

    def get_dataframe(
        self,
        *_,
        data_frequency: float,
        start_time: datetime | None = None,
        rng_seed: int | None = None,
        real_time: bool = True,
        end_time: datetime | None = None,
        record_count: int | None = None,
    ) -> pd.DataFrame:
        """
        Helper function to get real-time stream data in bulk within a dataframe.

        :param data_frequency: The rate at which to produce the data stream in records/second.
        :param start_time: Optional start time that the producer can start at.
        :param rng_seed: Optional seed number for random number generator
        :param real_time: Boolean flag denoting whether to simulate real-time stream or not.
        If false then data will be produced as fast as Python can do it.
        :param end_time: Optional timestamp at which to stop producing records
        :param record_count: Alternative optional means of stopping data production.
        Total number of records to produce.
        :returns: pandas DataFrame containing batch of real-time data
        """
        if end_time is None and record_count is None:
            raise ValueError("Either end_time or record_count must be specified.")

        if end_time is not None and end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        data: list[AccelerometerDataPoint] = []

        stream = self.generate_data_stream(
            data_frequency=data_frequency,
            start_time=start_time,
            rng_seed=rng_seed,
            real_time=real_time,
        )

        if end_time is None and record_count is not None:
            for rec in islice(stream, record_count):
                data.append(rec)
            return pd.DataFrame(data)

        for rec in stream:
            data.append(rec)
            if end_time is not None and rec["timestamp"] >= end_time:
                break
            if record_count is not None and len(data) >= record_count:
                break
        return pd.DataFrame(data)
