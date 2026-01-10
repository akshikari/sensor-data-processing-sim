# API Reference

## AccelerometerGenerator

The main entry point for simulating an accelerometer.

**Import:**

```python
from generators.accelerometer.generator import AccelerometerGenerator
```

### Constructor

```python
AccelerometerGenerator(
    id: UUID,
    generate_data_params: GenerateDataParams | None = None,
    anomaly_data_params: AnomalousDataModifierParams | None = None,
    anomaly_state: AnomalyState | None = None,
    stream_state: StreamState | None = None
)
```

### Methods

#### `get_dataframe`

Generates a batch of data as a Pandas DataFrame.

| Parameter        | Type    | Description                                                        |
| :--------------- | :------ | :----------------------------------------------------------------- |
| `data_frequency` | `float` | Sampling rate in Hz.                                               |
| `record_count`   | `int`   | Number of samples to generate.                                     |
| `real_time`      | `bool`  | If `True`, method sleeps to simulate elapsed time. Default `True`. |

#### `generate_data_stream`

Yields a generator of `AccelerometerDataPoint`.

| Parameter        | Type    | Description                            |
| :--------------- | :------ | :------------------------------------- |
| `data_frequency` | `float` | Sampling rate in Hz.                   |
| `real_time`      | `bool`  | If `True`, enforces wall-clock timing. |

---

## Configuration Models

**Import:**

```python
from generators.accelerometer.models import GenerateDataParams, AnomalousDataModifierParams
```

### GenerateDataParams

Controls the "normal" walking behavior.

| Field                 | Default | Description                                                       |
| :-------------------- | :------ | :---------------------------------------------------------------- |
| `gait_frequency_hz`   | `2.0`   | Step cadence (steps/sec). Controls primary oscillation frequency. |
| `amplitude_sway_m`    | `0.05`  | Lateral (Y-axis) displacement amplitude.                          |
| `amplitude_bounce_m`  | `0.02`  | Vertical (Z-axis) displacement amplitude.                         |
| `amplitude_roll_rad`  | `0.05`  | Roll rotation amplitude (side-to-side tilt).                      |
| `amplitude_pitch_rad` | `0.05`  | Pitch rotation amplitude (forward-back tilt).                     |
| `noise_std_dev`       | `0.05`  | Standard deviation of Gaussian noise added to signal.             |
| `gravity_mps2`        | `9.81`  | Gravitational constant ($g$).                                     |

### AnomalousDataModifierParams

Controls fault injection.

| Field               | Description                                                                  |
| :------------------ | :--------------------------------------------------------------------------- |
| `z_amp_modifier`    | Multiplier for Z-amplitude. Values < 1.0 simulate "limping" (weak push-off). |
| `time_drift_offset` | Adds cumulative time error per sample (simulates clock drift).               |
| `step_time_delay`   | Adds a delay (pause) at specific step intervals.                             |
| `step_frequency`    | Determines how often the anomaly occurs (e.g., every 4th step).              |
