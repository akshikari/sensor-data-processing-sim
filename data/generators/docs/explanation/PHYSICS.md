# Physics Implementation

This document details the mathematical model used in `AccelerometerGenerator._calculate_data_point` to simulate IMU readings.

## Core Philosophy

The generator does not use a rigid body dynamics engine (like PyBullet). Instead, it uses a **Kinematic Approximation** based on the observation that quadruped walking patterns are periodic and sinusoidal.

The robot's motion is modeled using **[Simple Harmonic Motion (SHM)](../../services/sensor_sim_api/docs/GLOSSARY.md#simple-harmonic-motion-shm)**.

## 1. Orientation (Attitude)

Before calculating linear acceleration, the robot's orientation in 3D space must be determined. As the robot walks, it "wobbles" in a predictable pattern.

### Roll & Pitch
Orientation angles ($\theta$) are modeled as sine waves:

$$ \theta(t) = \theta\_{base} + A \cdot \sin(\omega \cdot t + \phi) $$

- $\omega$ (Angular Frequency) is derived from the `gait_frequency_hz`.
- $A$ (Amplitude) is configurable (`amplitude_roll_rad`, `amplitude_pitch_rad`).

### Rotation Matrix
These Euler angles are converted into a **[Rotation Matrix](../services/sensor_sim_api/docs/GLOSSARY.md#rotation-matrix)** ($R_{bw}$) that transforms a vector from the Body Frame to the World Frame.

## 2. World Frame Acceleration

The robot's physical movement is calculated in the **[World Frame](../services/sensor_sim_api/docs/GLOSSARY.md#world-frame)**.

Since position is sinusoidal ($x(t) = A \sin(\omega t)$), acceleration is its second derivative:

$$ a(t) = \frac{d^2x}{dt^2} = -A \cdot \omega^2 \cdot \sin(\omega t) $$

This is computed for:
- **Y-axis (Sway)**: Lateral side-to-side motion.
- **Z-axis (Bounce)**: Vertical up-and-down motion.
- **X-axis (Surge)**: Currently assumed to be 0 (constant velocity).

## 3. Proper Acceleration (The Output)

An accelerometer measures **[Specific Force](../services/sensor_sim_api/docs/GLOSSARY.md#specific-force)** (Proper Acceleration), not coordinate acceleration. This means it measures the force _fighting_ gravity.

To get the final sensor reading in the **[Body Frame](../services/sensor_sim_api/docs/GLOSSARY.md#body-frame)**:

1.  Start with World Acceleration vector ($a_{world}$).
2.  Subtract the Gravity vector ($g = [0, 0, -9.81]$).
3.  Rotate the result into the Body Frame using the inverse rotation matrix ($R_{wb} = R_{bw}^T$).

$$ a*{body} = R*{bw}^T \cdot (a\_{world} - g) $$

## 4. Noise Injection

Finally, to make the data realistic, we add white Gaussian noise:

$$ a*{final} = a*{body} + \mathcal{N}(0, \sigma^2) $$

Where $\sigma$ is the `noise_std_dev` parameter.
