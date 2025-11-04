"""
Performance and real-time accuracy tests for AccelerometerGenerator.
Tests focus on throughput, latency, timing accuracy, and resource efficiency.
"""

import asyncio
import time
from datetime import datetime, timezone
from itertools import islice
from uuid import uuid4

import numpy as np
import pytest

from generators.accelerometer.generator import AccelerometerGenerator
from generators.accelerometer.models import (
    AnomalousDataModifierParams,
    GenerateDataParams,
)


class TestNonRealTimePerformance:
    """Test performance of non-real-time (fast) data generation"""

    @pytest.mark.parametrize(
        "frequency, duration, expected_samples",
        [
            pytest.param(100, 10, 1_000, id="1k_samples"),
            pytest.param(100, 100, 10_000, id="10k_samples"),
            pytest.param(1000, 100, 100_000, id="100k_samples"),
            pytest.param(1000, 1000, 1_000_000, id="1M_samples", marks=pytest.mark.slow),
        ],
    )
    def test_non_realtime_throughput(self, frequency: int, duration: int, expected_samples: int):
        """Test throughput for non-real-time data generation"""
        # Target: At least 70k samples/sec (reasonable for most systems)
        min_samples_per_sec = 70_000
        max_duration = expected_samples / min_samples_per_sec + 1.0  # Add 1.0s buffer

        sensor_id = uuid4()
        gen = AccelerometerGenerator(
            sensor_id=sensor_id,
            generate_data_params=GenerateDataParams(noise_std_dev=0.01),
        )

        start_time = time.perf_counter()
        df = gen.get_dataframe(
            data_frequency=frequency,
            record_count=expected_samples,
            real_time=False,
        )
        end_time = time.perf_counter()
        duration_sec = end_time - start_time

        assert len(df) == expected_samples

        throughput = expected_samples / duration_sec
        print(f"\n{expected_samples} samples: {duration_sec:.4f}s, {throughput:.0f} samples/sec")

        assert duration_sec < max_duration, (
            f"Generation of {expected_samples} samples took {duration_sec:.4f}s, "
            f"exceeding threshold of {max_duration:.4f}s "
            f"(target: {min_samples_per_sec:,} samples/sec)"
        )

    def test_memory_efficiency_large_batch(self):
        """Test memory efficiency when generating large batches"""
        sensor_id = uuid4()
        gen = AccelerometerGenerator(sensor_id=sensor_id)

        # Generate 100k samples using batch streaming to avoid large list accumulation
        batch_size = 1000
        sample_count = 100_000
        expected_batches = sample_count // batch_size

        batches = gen.batch_stream(
            data_frequency=1000,
            batch_size=batch_size,
            real_time=False,
        )

        batch_count = 0
        for batch in islice(batches, expected_batches):
            assert len(batch) == batch_size
            batch_count += 1

        gen.stop()
        assert batch_count == expected_batches

    @pytest.mark.parametrize(
        "frequency",
        [10, 50, 100, 500, 1000],
    )
    def test_generation_speed_vs_frequency(self, frequency: int):
        """Test that generation speed scales appropriately with frequency"""
        sensor_id = uuid4()
        gen = AccelerometerGenerator(sensor_id=sensor_id)

        sample_count = 10_000
        start_time = time.perf_counter()
        df = gen.get_dataframe(
            data_frequency=frequency,
            record_count=sample_count,
            real_time=False,
        )
        duration = time.perf_counter() - start_time

        assert len(df) == sample_count
        throughput = sample_count / duration
        print(f"\nFrequency {frequency}Hz: {throughput:.0f} samples/sec")

        # Should maintain high throughput regardless of frequency
        assert throughput > 50_000  # At least 50k samples/sec


class TestRealTimeTiming:
    """Test timing accuracy of real-time streaming"""

    def test_realtime_timing_accuracy_low_frequency(self):
        """Test timing accuracy at low frequency (10 Hz)"""
        sensor_id = uuid4()
        frequency = 10
        sample_count = 20  # 2 seconds of data

        gen = AccelerometerGenerator(sensor_id=sensor_id)

        start_wall = time.perf_counter()
        df = gen.get_dataframe(
            data_frequency=frequency,
            record_count=sample_count,
            real_time=True,
        )
        elapsed_wall = time.perf_counter() - start_wall

        expected_duration = (sample_count - 1) / frequency
        timing_tolerance = 0.05  # 50ms tolerance

        assert len(df) == sample_count
        assert elapsed_wall == pytest.approx(expected_duration, abs=timing_tolerance)

    def test_realtime_timing_accuracy_high_frequency(self):
        """Test timing accuracy at high frequency (100 Hz)"""
        sensor_id = uuid4()
        frequency = 100
        sample_count = 200  # 2 seconds of data

        gen = AccelerometerGenerator(sensor_id=sensor_id)

        start_wall = time.perf_counter()
        df = gen.get_dataframe(
            data_frequency=frequency,
            record_count=sample_count,
            real_time=True,
        )
        elapsed_wall = time.perf_counter() - start_wall

        expected_duration = (sample_count - 1) / frequency
        timing_tolerance = 0.05  # 50ms tolerance

        assert len(df) == sample_count
        assert elapsed_wall == pytest.approx(expected_duration, abs=timing_tolerance)

    def test_realtime_vs_non_realtime_speed(self):
        """Verify that non-real-time mode is significantly faster than real-time"""
        sensor_id = uuid4()
        frequency = 50
        sample_count = 100

        # Real-time generation
        gen_rt = AccelerometerGenerator(sensor_id=sensor_id)
        start_rt = time.perf_counter()
        df_rt = gen_rt.get_dataframe(
            data_frequency=frequency,
            record_count=sample_count,
            real_time=True,
        )
        duration_rt = time.perf_counter() - start_rt

        # Non-real-time generation
        gen_nrt = AccelerometerGenerator(sensor_id=sensor_id)
        start_nrt = time.perf_counter()
        df_nrt = gen_nrt.get_dataframe(
            data_frequency=frequency,
            record_count=sample_count,
            real_time=False,
        )
        duration_nrt = time.perf_counter() - start_nrt

        assert len(df_rt) == sample_count
        assert len(df_nrt) == sample_count

        # Non-real-time should be at least 10x faster
        speedup = duration_rt / duration_nrt
        print(f"\nReal-time: {duration_rt:.4f}s, Non-real-time: {duration_nrt:.4f}s, Speedup: {speedup:.1f}x")
        assert speedup > 10

    def test_timestamp_precision_realtime(self):
        """Test timestamp precision in real-time mode"""
        sensor_id = uuid4()
        frequency = 50  # Lower frequency for more stable timing
        gen = AccelerometerGenerator(sensor_id=sensor_id)

        df = gen.get_dataframe(
            data_frequency=frequency,
            record_count=10,
            real_time=True,
        )

        # Check that timestamp intervals match expected frequency
        timestamps = df["timestamp"].values
        time_diffs = np.diff(timestamps).astype('timedelta64[ns]').astype(np.float64) / 1e9

        expected_interval = 1.0 / frequency
        # Real-time mode should have fairly consistent intervals
        # Allow 20ms tolerance per interval due to system scheduling
        for diff in time_diffs:
            assert abs(diff - expected_interval) < 0.02


class TestStreamContinuity:
    """Test that streams maintain continuity and consistency"""

    def test_continuous_sequence_numbers(self):
        """Test that sequence numbers are continuous without gaps"""
        sensor_id = uuid4()
        gen = AccelerometerGenerator(sensor_id=sensor_id)

        df = gen.get_dataframe(
            data_frequency=100,
            record_count=1000,
            real_time=False,
        )

        # Check sequence continuity
        expected_sequence = np.arange(1000)
        assert np.array_equal(df["sequence"].values, expected_sequence)

    def test_timestamp_monotonicity(self):
        """Test that timestamps are strictly monotonically increasing"""
        sensor_id = uuid4()
        gen = AccelerometerGenerator(sensor_id=sensor_id)

        df = gen.get_dataframe(
            data_frequency=50,
            record_count=500,
            real_time=False,
        )

        timestamps = df["timestamp"].values
        timestamp_diffs = np.diff(timestamps.astype('int64'))

        # All differences should be positive (strictly increasing)
        assert np.all(timestamp_diffs > 0)

    def test_no_duplicate_timestamps(self):
        """Test that no duplicate timestamps are generated"""
        sensor_id = uuid4()
        gen = AccelerometerGenerator(sensor_id=sensor_id)

        df = gen.get_dataframe(
            data_frequency=100,
            record_count=1000,
            real_time=False,
        )

        # Check for uniqueness
        assert df["timestamp"].nunique() == len(df)


class TestDataQualityWithAnomalies:
    """Test data quality and accuracy with anomalous parameters"""

    def test_anomaly_impact_on_performance(self):
        """Test that anomalies don't significantly degrade performance"""
        sensor_id = uuid4()
        sample_count = 10_000

        # Normal generation
        gen_normal = AccelerometerGenerator(sensor_id=sensor_id)
        start_normal = time.perf_counter()
        df_normal = gen_normal.get_dataframe(
            data_frequency=100,
            record_count=sample_count,
            real_time=False,
        )
        duration_normal = time.perf_counter() - start_normal

        # Generation with all anomaly types
        gen_anomaly = AccelerometerGenerator(
            sensor_id=sensor_id,
            anomaly_data_params=AnomalousDataModifierParams(
                z_amp_modifier=0.5,
                time_drift_offset=0.001,
                step_time_delay=0.01,
                step_frequency=4,
            ),
        )
        start_anomaly = time.perf_counter()
        df_anomaly = gen_anomaly.get_dataframe(
            data_frequency=100,
            record_count=sample_count,
            real_time=False,
        )
        duration_anomaly = time.perf_counter() - start_anomaly

        assert len(df_normal) == sample_count
        assert len(df_anomaly) == sample_count

        # Anomaly generation should not be more than 2x slower
        slowdown = duration_anomaly / duration_normal
        print(f"\nNormal: {duration_normal:.4f}s, Anomalous: {duration_anomaly:.4f}s, Slowdown: {slowdown:.2f}x")
        assert slowdown < 2.0

    def test_cumulative_drift_accuracy(self):
        """Test accuracy of cumulative time drift calculations"""
        sensor_id = uuid4()
        time_drift = 0.001  # 1ms per sample
        sample_count = 1000

        gen = AccelerometerGenerator(
            sensor_id=sensor_id,
            generate_data_params=GenerateDataParams(noise_std_dev=0.0),
            anomaly_data_params=AnomalousDataModifierParams(
                time_drift_offset=time_drift,
            ),
        )

        df = gen.get_dataframe(
            data_frequency=50,
            record_count=sample_count,
            real_time=False,
        )

        # Check that cumulative drift matches expected
        assert gen.anomaly_state.cumulative_time_drift == pytest.approx(
            (sample_count - 1) * time_drift,
            abs=1e-9,
        )

    def test_step_delay_accuracy(self):
        """Test accuracy of intermittent step delay calculations"""
        sensor_id = uuid4()
        step_delay = 0.05  # 50ms
        step_frequency = 4

        gen = AccelerometerGenerator(
            sensor_id=sensor_id,
            generate_data_params=GenerateDataParams(
                noise_std_dev=0.0,
                gait_frequency_hz=2.0,  # 2 steps per second
            ),
            anomaly_data_params=AnomalousDataModifierParams(
                step_time_delay=step_delay,
                step_frequency=step_frequency,
            ),
        )

        df = gen.get_dataframe(
            data_frequency=50,
            record_count=1000,
            real_time=False,
        )

        # Calculate expected number of delayed steps
        # At 2 Hz gait and 20 seconds (1000 samples / 50 Hz), we have ~40 steps
        # Every 4th step is delayed, so ~10 delays
        expected_total_delay = gen.anomaly_state.cumulative_step_delay
        print(f"\nTotal accumulated step delay: {expected_total_delay:.4f}s")

        # Should have some accumulated delay
        assert expected_total_delay > 0


class TestAsyncPerformance:
    """Test performance of async streaming operations"""

    @pytest.mark.anyio
    async def test_async_stream_throughput(self):
        """Test throughput of async stream generation"""
        sensor_id = uuid4()
        gen = AccelerometerGenerator(sensor_id=sensor_id)

        sample_count = 10_000
        start_time = time.perf_counter()

        stream = gen.async_generate_data_stream(
            data_frequency=1000,
            real_time=False,
        )

        count = 0
        async for _ in stream:
            count += 1
            if count >= sample_count:
                break

        gen.stop()
        duration = time.perf_counter() - start_time

        throughput = sample_count / duration
        print(f"\nAsync stream: {sample_count} samples in {duration:.4f}s ({throughput:.0f} samples/sec)")

        # Should maintain high throughput
        assert throughput > 50_000  # At least 50k samples/sec

    @pytest.mark.anyio
    async def test_async_batch_overhead(self):
        """Test overhead of async batch streaming vs regular batching"""
        sensor_id = uuid4()
        batch_size = 100
        batch_count = 100

        # Sync batch stream
        gen_sync = AccelerometerGenerator(sensor_id=sensor_id)
        start_sync = time.perf_counter()
        batches_sync = gen_sync.batch_stream(
            data_frequency=1000,
            batch_size=batch_size,
            real_time=False,
        )
        list(islice(batches_sync, batch_count))
        gen_sync.stop()
        duration_sync = time.perf_counter() - start_sync

        # Async batch stream
        gen_async = AccelerometerGenerator(sensor_id=sensor_id)
        start_async = time.perf_counter()
        batches_async = gen_async.async_batch_stream(
            data_frequency=1000,
            batch_size=batch_size,
            real_time=False,
        )
        count = 0
        async for _ in batches_async:
            count += 1
            if count >= batch_count:
                break
        gen_async.stop()
        duration_async = time.perf_counter() - start_async

        print(f"\nSync: {duration_sync:.4f}s, Async: {duration_async:.4f}s")

        # Async overhead should be minimal (< 50% slower)
        assert duration_async < duration_sync * 1.5


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_high_frequency(self):
        """Test generation at very high frequency (10 kHz)"""
        sensor_id = uuid4()
        gen = AccelerometerGenerator(sensor_id=sensor_id)

        frequency = 10_000
        sample_count = 10_000  # 1 second of data

        start_time = time.perf_counter()
        df = gen.get_dataframe(
            data_frequency=frequency,
            record_count=sample_count,
            real_time=False,
        )
        duration = time.perf_counter() - start_time

        assert len(df) == sample_count

        # Check timestamp precision - should be properly spaced by 1/frequency
        expected_delta = 1.0 / frequency
        actual_deltas = df["timestamp"].diff().dt.total_seconds().dropna()
        assert np.allclose(actual_deltas, expected_delta, atol=1e-9)

    def test_very_low_frequency(self):
        """Test generation at very low frequency (0.1 Hz)"""
        sensor_id = uuid4()
        gen = AccelerometerGenerator(sensor_id=sensor_id)

        frequency = 0.1
        sample_count = 5  # 50 seconds of data, but we'll use non-realtime

        df = gen.get_dataframe(
            data_frequency=frequency,
            record_count=sample_count,
            real_time=False,
        )

        assert len(df) == sample_count

        # Check timestamp intervals - should be 10 seconds apart
        expected_delta = 1.0 / frequency  # 10 seconds
        actual_deltas = df["timestamp"].diff().dt.total_seconds().dropna()
        assert np.allclose(actual_deltas, expected_delta, atol=1e-9)

    def test_single_sample_generation(self):
        """Test generating a single sample"""
        sensor_id = uuid4()
        gen = AccelerometerGenerator(sensor_id=sensor_id)

        df = gen.get_dataframe(
            data_frequency=10,
            record_count=1,
            real_time=False,
        )

        assert len(df) == 1
        assert df["sequence"].iloc[0] == 0

    def test_fractional_frequency(self):
        """Test generation with fractional frequency"""
        sensor_id = uuid4()
        gen = AccelerometerGenerator(sensor_id=sensor_id)

        frequency = 33.33  # Approximately 30 Hz
        sample_count = 100

        df = gen.get_dataframe(
            data_frequency=frequency,
            record_count=sample_count,
            real_time=False,
        )

        assert len(df) == sample_count

        # Check timestamp precision with fractional frequency
        expected_delta = 1.0 / frequency
        actual_deltas = df["timestamp"].diff().dt.total_seconds().dropna()
        assert np.allclose(actual_deltas, expected_delta, atol=1e-9)
