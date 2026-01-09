# Sensor Data Processing Simulator

## Motivation & Vision

This project is to serve as a playground for learning many facets of software development and engineering I've always wanted to explore.
I've had a long-running passion for ML/AI and its applications in Robotics, as well as software product development.
As I explore the ML/AI and Robotics field via several courses and texts, I will be putting those skills into practice with this code
with the mindset of developing a robust, production-ready "product" around the creation, processing, and management of that data.
I wanted first start by getting as far as I can applying my current knowledge, then as I learn new concepts (front-end dev, ML/AI, Robotics development)
I will grow this project and apply those concepts. I also wanted to use this as an opportunity to learn and experiment with new tools I have always wanted
to explore, as well as improve my expertise with familiar tools

The goal is to build an end-to-end system starting from first principles:

1.  **Physics**: How does an accelerometer actually work? What is "Specific Force"?
2.  **Simulation**: How to procedurally generate realistic, noisy sensor data from scratch?
3.  **Service**: How to manage several simulated sensors via a modern API?
4.  **Streaming**: How to handle high-frequency data ingestion? (Future)
5.  **Visualization**: How to build a real-time dashboard? (Future)
6.  **Intelligence**: How to detect anomalies (limping, drift) using ML? (Future)

## Roadmap & Status

| Phase | Component          | Status     | Description                                                                            |
| :---- | :----------------- | :--------- | :------------------------------------------------------------------------------------- |
| **1** | **Physics Engine** | Complete\* | A Python library (`data/generators`) simulating IMU physics.                           |
| **2** | **Control Plane**  | Active     | A REST API (`services/sensor_sim_api`) to manage sensor lifecycles and configurations. |
| **3** | **Data Pipeline**  | Paused     | Batch/Stream processing (`transformers`, `writers`). Paused to focus on the API layer. |
| **4** | **Streaming**      | Planned    | Real-time data ingestion and processing.                                               |
| **5** | **Dashboard**      | Planned    | Web UI for visualizing live sensor streams.                                            |
| **6** | **ML/AI**          | Planned    | Real-time anomaly detection service.                                                   |

\* I have more or less completed a physics model for simulating real-time accelerometer data. Time-permitting I have plans to implement gyroscope
and magnetometer sensor simulators as well.

## Core Components

### 1. The Physics Engine (`data/generators`)

The heart of the simulator. It doesn't just play back static files; it calculates motion vectors in real-time.

- **Model**: Uses [Simple Harmonic Motion](services/sensor_sim_api/docs/GLOSSARY.md#simple-harmonic-motion-shm) to approximate quadruped walking gaits.
- **Features**: Simulates [Body Frame](services/sensor_sim_api/docs/GLOSSARY.md#body-frame) rotation, gravity subtraction, and white noise injection.

### 2. The Control Plane (`services/sensor_sim_api`)

A modern FastAPI service to provision and manage sensors.

- **Docs**: **[Full Documentation Hub](services/sensor_sim_api/docs/INDEX.md)**
- **Features**: Sensor CRUD, configuration management, SQLite/PostgreSQL storage.

### 3. Legacy/Paused Components

The `data/transformers`, `data/writers`, and `data/streams` directories contain early experiments with batch processing and S3 writers. These are currently frozen as the architecture evolves toward a real-time microservices approach.

## Technology Stack

- **Languages**: Python 3.12+ (Type-hinted, Pydantic v2)
- **Frameworks**: FastAPI, NumPy, Pandas, SciPy
- **Infrastructure**: Docker, Dagger, Terraform (AWS)
- **Tooling**: UV (Package management), Ruff (Linting), Make

## Getting Started

The best place to start exploring is the **Sensor Simulation API**.

1.  **[Read the Quickstart Guide](services/sensor_sim_api/docs/getting-started/QUICKSTART.md)**
2.  **[Create Your First Sensor](services/sensor_sim_api/docs/getting-started/FIRST-SENSOR.md)**
3.  **[Browse the Glossary](services/sensor_sim_api/docs/GLOSSARY.md)** to understand the physics terms.

## Documentation

Detailed documentation is currently housed within the API service:

- **[Project Index](services/sensor_sim_api/docs/INDEX.md)**
- **[Architecture](services/sensor_sim_api/docs/reference/ARCHITECTURE.md)**
- **[Physics Data Model](services/sensor_sim_api/docs/explanation/DATA-MODEL.md)**
- **[Troubleshooting](services/sensor_sim_api/TROUBLESHOOTING.md)**

---
