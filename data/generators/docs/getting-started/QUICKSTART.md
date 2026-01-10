# Quickstart Guide

This guide covers the three main ways to use the `generators` library: streaming, async streaming, and batch generation.

## 1. Batch Generation (Pandas DataFrame)

The most common use case for data science and analysis is generating a static dataset.

```python
from uuid import uuid4
from generators.accelerometer.generator import AccelerometerGenerator

# Initialize
gen = AccelerometerGenerator(id=uuid4())

# Generate 10 seconds of data at 50Hz
df = gen.get_dataframe(
    data_frequency=50.0,
    record_count=500
)

# Result is a standard Pandas DataFrame
print(df.describe())
```

## 2. Real-Time Streaming

Simulate a live sensor feed. This method yields data points one by one, respecting wall-clock time (it will sleep between yields to match the frequency).

```python
from uuid import uuid4
from generators.accelerometer.generator import AccelerometerGenerator

gen = AccelerometerGenerator(id=uuid4())

# Stream data at 10Hz (will print 10 lines per second)
stream = gen.generate_data_stream(
    data_frequency=10.0,
    real_time=True
)

try:
    for point in stream:
        print(f"Timestamp: {point['timestamp']} | Z-Accel: {point['accel_z']:.4f}")
except KeyboardInterrupt:
    gen.stop()
```

## 3. High-Throughput Streaming (Non-Realtime)

If you need to generate a massive amount of data for a stress test or backfill, disable `real_time`.

```python
stream = gen.generate_data_stream(
    data_frequency=100.0,
    real_time=False  # Generates as fast as CPU allows
)
```

## 4. Async Streaming

For integration with async applications (like FastAPI).

```python
import asyncio
from uuid import uuid4
from generators.accelerometer.generator import AccelerometerGenerator

async def run_sensor():
    gen = AccelerometerGenerator(id=uuid4())

    async for point in gen.async_generate_data_stream(data_frequency=10.0):
        await send_to_client(point)

asyncio.run(run_sensor())
```

## 5. Configuring Physics

Customize the simulation using `GenerateDataParams`.

```python
from generators.accelerometer.models import GenerateDataParams

custom_params = GenerateDataParams(
    gait_frequency_hz=1.5,       # Slower walk
    amplitude_bounce_m=0.1,      # Higher bounce
    noise_std_dev=0.0            # Perfect signal (no noise)
)

gen = AccelerometerGenerator(id=uuid4(), generate_data_params=custom_params)
```
