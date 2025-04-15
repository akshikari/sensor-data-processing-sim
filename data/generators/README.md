# Data Generators

A package focused on generating mock data for specific use cases.
Initially this will focus on generating mock robotics sensor data

## Available Mock Data Generators

### Accelerometer Data

- This data is generated using some base assumptions:
  - It is a quadruped robot walking following a sinusoidal walking pattern, thus we can use Simple Harmonic Motion to model its movement
  - The robot is walking in a straight line on a flat, horizontal surface
  - To simulate noisy data we add a `noise_std_dev` parameter to modify the Gaussian distribution of the acceleration data
