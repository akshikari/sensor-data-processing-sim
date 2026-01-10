# Data Model & Physics

The Simulator does not play back pre-recorded data. Instead, it generates data procedurally using physics-based statistical models.

> For a deep dive into the mathematical implementation, see the **[Generators Physics Documentation](../../../../data/generators/docs/explanation/PHYSICS.md)**.

## Accelerometer Model

The accelerometer simulation is based on **[Simple Harmonic Motion (SHM)](../GLOSSARY.md#simple-harmonic-motion-shm)** to approximate quadruped walking patterns.

### Coordinate Frames

- **[World Frame](../GLOSSARY.md#world-frame)**: Static reference frame. Gravity points down (-Z).
- **[Body Frame](../GLOSSARY.md#body-frame)**: Attached to the sensor/robot. This is what the accelerometer measures.

### Motion Logic

The simulation calculates linear acceleration in 3 steps:

1.  **Orientation (Attitude):**
    The "wobble" of a walking robot is simulated using sinusoidal waves for [Roll](../GLOSSARY.md#roll-phi) and [Pitch](../GLOSSARY.md#pitch-theta).
    $$ \theta(t) = \theta\_{base} + A \cdot \sin(\omega \cdot t + \phi) $$

2.  **World Acceleration:**
    Then the 2nd derivative of position (acceleration) for [sway](../GLOSSARY.md#sway) (Y-axis) and [bounce](../GLOSSARY.md#bounce) (Z-axis) is calculated.
    $$ a(t) = -A \cdot \omega^2 \cdot \sin(\omega \cdot t + \phi) $$

3.  **Body Acceleration:**
    The final step is to rotate the World Acceleration vector into the Body Frame and subtract [gravity](../GLOSSARY.md#gravity-vector-g) (since accelerometers measure [proper acceleration](../GLOSSARY.md#proper-acceleration)).
    $$ a*{body} = R*{wb} (a\_{world} - g) $$

### Noise Model

Finally, white Gaussian noise is added to simulate sensor imperfection.

- **Distribution**: Normal (Gaussian)
- **Mean**: 0.0
- **Std Dev**: Configurable via `noise_std_dev`

### Anomalies

Anomalies can be injected into the stream to test fault-detection algorithms:

- **Z-Amplitude Modifier**: Dampens or amplifies vertical motion (e.g., limping).
- **Step Frequency**: Inserts anomalies at specific step intervals.
- **Time Drift**: Simulates clock skew.

## Extensibility

The `generators` library is designed to be extensible. Time permitting, new sensor types ([Gyroscope](../GLOSSARY.md#gyroscope), [Magnetometer](../GLOSSARY.md#magnetometers))
will be added by implementing the generator interface and defining their own physics models.
