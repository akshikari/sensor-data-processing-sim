# Generators Library

**A high-performance physics simulation library for robotics sensor data.**

This Python package provides the core simulation engines used to generate realistic, noisy sensor streams (IMU data) based on kinematic models.

## Overview

The library allows developers to:

- Generate continuous streams of accelerometer data.
- Simulate realistic quadruped walking gaits using Simple Harmonic Motion (SHM).
- Inject anomalies (drifts, delays, signal amplitude changes) for testing purposes.
- Export data directly to Pandas DataFrames for analysis.

## Quick Links

- **[Documentation Index](docs/INDEX.md)**
- **[API Reference](docs/reference/API.md)**
- **[Physics Explanation](docs/explanation/PHYSICS.md)**

## Installation

This package is managed via `uv`.

```bash
uv add generators
```

## Basic Usage

```python
from uuid import uuid4
from generators.accelerometer.generator import AccelerometerGenerator
from generators.accelerometer.models import GenerateDataParams

# 1. Configure the simulation
params = GenerateDataParams(gait_frequency_hz=2.0, noise_std_dev=0.01)

# 2. Initialize the generator
gen = AccelerometerGenerator(id=uuid4(), generate_data_params=params)

# 3. Generate a batch of data (DataFrame)
df = gen.get_dataframe(data_frequency=100.0, record_count=1000)
print(df.head())
```

## Testing

Unit tests ensure the statistical properties of the generated data match the configured parameters (e.g., verifying the output frequency matches the input gait frequency).

```bash
make test-generators
```
