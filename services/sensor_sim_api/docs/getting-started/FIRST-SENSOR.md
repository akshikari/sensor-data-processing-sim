# Creating Your First Sensor

This tutorial will guide you through creating and simulating your first accelerometer sensor using the API.

## Prerequisites

- Application running (`make dev`)
- `curl` installed (or use the Swagger UI at `http://localhost:8000/docs`)

## Step 1: Create a Sensor Type

First, verify that the "accelerometer" sensor type exists.

**Request:**

```bash
curl -X 'GET' \
  'http://localhost:8000/api/v1/sensor-type/{sensor-type-uuid}' \
  -H 'accept: application/json'
```

If it returns 404, create it:

**Request:**

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/sensor-type/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "id": "{sensor-type-uuid}",
  "name": "accelerometer"
}'
```

**Response:**

```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "name": "accelerometer"
}
```

## Step 2: Create an Accelerometer

Now you can create a specific [accelerometer](../GLOSSARY.md#accelerometer) instance. Can give it a standard walking [gait frequency](../GLOSSARY.md#gait) (2 Hz).

!> [!NOTE]

> Please read the documentation on [GenerateDataParams](../../../../data/generators/docs/reference/API.md#generatedataparams). A lot of these parameters have defaults worth knowing.

**Request:**

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/sensors/accelerometer/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sensor_type_id": "123e4567-e89b-12d3-a456-426614174000",
  "generate_data_params": {
    "gait_frequency_hz": 2.0,
    "amplitude_sway_m": 0.05,
    "noise_std_dev": 0.01
  }
}'
```

**Response:**
You will receive a JSON object with a new `id` (UUID).

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "sensor_type_id": "123e4567-e89b-12d3-a456-426614174000",
  "generate_data_params": {
    "gait_frequency_hz": 2.0,
    ...
  },
  ...
}
```

_Note: Save this ID for the next steps._

## Step 3: Retrieve Sensor Details

Verify your sensor is stored correctly.

**Request:**

```bash
curl -X 'GET' \
  'http://localhost:8000/api/v1/sensors/accelerometer/550e8400-e29b-41d4-a716-446655440000' \
  -H 'accept: application/json'
```

## Step 4: Update Configuration

Let's make the sensor simulate a faster walk (running).

**Request:**

```bash
curl -X 'PATCH' \
  'http://localhost:8000/api/v1/sensors/accelerometer/550e8400-e29b-41d4-a716-446655440000' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "generate_data_params": {
    "gait_frequency_hz": 3.5
  }
}'
```

## Summary

You have successfully:

1. Defined a Sensor Type.
2. Provisioned a virtual Accelerometer.
3. Updated its simulation parameters.

In a real deployment, a separate Streaming Service would pick up these configurations to generate the high-frequency data stream.
