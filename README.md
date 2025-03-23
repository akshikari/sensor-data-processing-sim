# Robotics Sensor Data Processing Simulation

Small side project to brush up on as well as learn some new DE skills by
processing simulated robotics sensor data.
Some important details:

- I will be creating two "versions" of some components like data generation
  - One version will simulate a real-time environment. Components can be run
    as services
  - The second version will have everything scheduled via Airflow. Since this
    is just a simulation I don't want to incur the costs of keeping things
    constantly running

Written using organic, free-range, GMO-free and LLM-free code
because \<Space> + \<Tab> ain't learnin'

## Project Brief

Create several components around creating, ingesting, and processing
sensor data. For the time being the data will be generated mock data.

## Components

### Data Simulation

For now, just some Python script(s) that will generate some
time-series data and publish to Kafka. Will containerize this
and push to Cloud Run and use Cloud Scheduler.

### Data Ingestion

Some Kafka listeners that will take the data and write to DB or GCS.
Again will have variants for real-time env sim and just batch jobs

### Data Processing

A number of python clean up and potentially some pyspark jobs to process
the data

### Orchestration

Just airflow

### Infrastructure

The cloud (GCP)

### Deployment

Terraform
