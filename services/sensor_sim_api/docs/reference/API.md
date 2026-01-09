# API Reference

The Sensor Simulation API follows RESTful principles.

- **Base URL**: `http://localhost:8000` (development)
- **Documentation**: `http://localhost:8000/docs` (Swagger UI)

## Accelerometers

Manage virtual [accelerometer](../GLOSSARY.md#accelerometer) sensors.

| Method   | Endpoint              | Description                                 |
| :------- | :-------------------- | :------------------------------------------ |
| `GET`    | `/accelerometer/{id}` | Retrieve sensor configuration and state.    |
| `POST`   | `/accelerometer/`     | Create a new accelerometer sensor.          |
| `PATCH`  | `/accelerometer/{id}` | Update parameters (e.g., frequency, noise). |
| `DELETE` | `/accelerometer/{id}` | Archive (soft delete) a sensor.             |

### Resource Model

**Accelerometer**

```json
{
  "id": "uuid",
  "sensor_type_id": "uuid",
  "generate_data_params": {
    "gait_frequency_hz": 2.0,
    "amplitude_sway_m": 0.05,
    "amplitude_bounce_m": 0.02,
    "noise_std_dev": 0.01
  },
  "anomalous_data_params": {
    "z_amp_modifier": 0.8,
    "step_frequency": 4
  },
  "stream_state": null,
  "create_ts": "timestamp",
  "update_ts": "timestamp"
}
```

## Sensor Types

Manage the definitions of sensor categories.

| Method   | Endpoint            | Description                   |
| :------- | :------------------ | :---------------------------- |
| `GET`    | `/sensor-type/{id}` | Retrieve sensor type details. |
| `POST`   | `/sensor-type/`     | Define a new sensor type.     |
| `PATCH`  | `/sensor-type/{id}` | Update sensor type name.      |
| `DELETE` | `/sensor-type/{id}` | Archive a sensor type.        |

### Resource Model

**SensorType**

```json
{
  "id": "uuid",
  "name": "string"
}
```

## Error Handling

Standard HTTP status codes are used.

| Code  | Meaning                               |
| :---- | :------------------------------------ |
| `200` | Success                               |
| `201` | Created                               |
| `204` | No Content (Success)                  |
| `400` | Bad Request                           |
| `404` | Not Found                             |
| `409` | Conflict (Duplicate ID)               |
| `422` | Validation Error (Invalid payload)    |
| `500` | Internal Server Error                 |
| `503` | Service Unavailable (Database issues) |
