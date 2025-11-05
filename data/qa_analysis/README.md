# Quality Analysis Notebooks

This directory contains Jupyter notebooks used for exploratory data analysis and quality assurance of generated sensor data.

## Running Notebooks

### Setup Jupyter Kernel with uv

```bash
# From project root, sync dev dependencies (includes ipykernel)
uv sync --group dev

# Install the kernel spec
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=sensor-data-sim

# Start Jupyter Lab
uv run jupyter lab
```

### Select the Kernel

Once Jupyter Lab opens:
1. Open your notebook
2. Click on the kernel name in the top-right corner
3. Select "sensor-data-sim" from the list

### Alternative: Run without kernel installation

```bash
# Just run Jupyter Lab directly (uses current environment)
uv run jupyter lab
```

## Current Notebooks

- **accelerometer_gen_data_qa.ipynb**: Quality analysis of accelerometer data generation, validating physical accuracy and statistical properties of generated sensor data.
