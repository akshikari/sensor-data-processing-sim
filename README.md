# Robotics Sensor Data Processing Simulation

Small side project to brush up on as well as learn some new DE skills by
processing simulated robotics sensor data.
Some important details:

- I will be creating two "versions" of some components like data generation
  - One version will simulate a real-time environment. Components can be run
    as services.
  - The second version will have everything scheduled via Airflow. Since this
    is just a simulation I don't want to incur the costs of keeping things
    constantly running.
  - This will be a phased approach with the real-time component coming later.

## Project Brief

Create several components around creating, ingesting, and processing
sensor data. For the time being the data will be generated mock data.

## Components

### Data Simulation

For now, just some Python script(s) that will generate some mock sensor data.

### Data Ingestion

Taking a phased approach. Phase 1 will be just dumping the raw data into S3, then to Timestream.

### Data Processing

A number of python clean up and potentially some pyspark jobs to process the data

### Orchestration

Good ol' Apache Airflow. Looks like v3 came out relatively recently

### Infrastructure

- Storage:
  - AWS S3
  - AWS Timestream (WIP)
- Workflow Orchestration (WIP)
  - AWS MWAA (AWS Managed Workflows for Apache Airflow), using DockerOperator/KubernetesPodOperator
    - Or maybe just deploy my own Airflow. Latest version available via MWAA is 2.10.1
- Data Pipelines (WIP)
  - Custom framework
- Data Analytics/Visualization (WIP)

### Deployment

Terraform

## New Tools Learned for this Project

- Terraform
- AWS (new but similar to GCP)
- Apache Airflow
- Apache Parquet
- Apache Arrow
- NumPy/SciPy
- Pandas
- Great Expectations
- uv
