# Robotics Sensor Data Processing Simulation

This project is a personal initiative to design and implement an end-to-end data pipeline for simulating, ingesting, processing, and analyzing robotics sensor data. The primary goal is to deepen existing data engineering skills and learn new technologies relevant to modern data-intensive applications, particularly in the context of robotics and AI.

The project follows a phased development approach, starting with foundational components for data generation and storage, with plans to incorporate more advanced processing, orchestration, and data quality frameworks.

## Current Status & Features Implemented (Phase 1)

The initial phase focuses on establishing a core pipeline to generate mock accelerometer data for a simulated walking quadruped and land this data into AWS S3.

**Key Components & Features:**

1.  **Data Generation (`data/generators` package):**

    - **Accelerometer Simulation:** Implemented a Python-based accelerometer data generator (`generators.accelerometer.generate_data`) using a kinematic model. This model simulates the body motion of a walking quadruped (bounce, sway, pitch, roll) based on Simple Harmonic Motion (SHM) principles.
    - **Configurable Parameters:** Simulation parameters (gait frequency, speed, oscillation amplitudes, noise levels, etc.) are configurable via a Pydantic dataclass (`GenerateDataParams`).
    - **Efficient Output:** The generator is optimized to output simulated data directly as a Pandas DataFrame for efficient downstream processing.
    - **Technologies:** Python, NumPy, SciPy (for 3D rotations), Pydantic.

2.  **Data Storage Writer (`data/writers` package):**

    - **S3 Writer:** Developed an `S3Writer` class (`writers.s3_writer.S3Writer`) using `boto3` for robust interaction with AWS S3.
    - **Flexible Output Formats:** Supports writing data batches (received as `list[dict]` or `pd.DataFrame`) to S3 as either **Parquet** (default, columnar) or **CSV** files.
    - **Hive-Style Partitioning:** Implements dynamic, Hive-style partitioning (e.g., `year=YYYY/month=MM/day=DD/hour=HH/sensor_id=ID/`) based on configurable columns, enabling efficient querying for analytics.
    - **AWS Authentication:** Supports standard AWS credential chain, including explicit keys or IAM role assumption via STS.
    - **Technologies:** Python, Pandas, PyArrow (for Parquet), `boto3`.

3.  **Pipeline Orchestration (`data/streams` package):**

    - **Initial Pipeline Script:** A Python script (`streams.pipe_accelerometer_to_s3.run_pipeline`) orchestrates the current data flow:
      - Reads pipeline configuration (source generator parameters, S3 destination details). Eventually this will be from a YAML file or config DB.
      - Invokes the accelerometer data generator.
      - Performs necessary data transformations (e.g., deriving partition key values like year, month, day, hour from timestamps; converting UUIDs to strings for partitioning).
      - Initializes and calls the `S3Writer` to persist the data to S3.
    - **Configuration:** Uses a Pydantic dataclass (`PipelineConfigs`) for structuring pipeline settings.
    - **Technologies:** Python, Pydantic.

4.  **Infrastructure as Code (`infra/terraform`):**

    - **AWS Resource Provisioning:** Terraform scripts are used to define and provision the necessary AWS cloud infrastructure:
      - An S3 bucket configured as a data landing zone (with versioning, server-side encryption, and public access blocked).
      - An AWS Timestream database and table (as a planned target for future data loading).
      - Appropriate IAM Roles and Policies to grant necessary permissions (e.g., S3 write access for the application role, Timestream write access) following the principle of least privilege.
    - **Technologies:** Terraform, HCL, AWS (S3, Timestream, IAM).

5.  **Testing & QA:**

    - **Unit Tests:** Comprehensive unit tests written using `pytest` for:
      - The `generators` package (validating input, output structure, deterministic behavior, statistical properties of noise, timestamp correctness, and performance).
      - The `writers` package (mocking S3 interactions with `moto` to test initialization, partition path logic, successful Parquet/CSV writes, content verification, and error handling).
      - The `streams` pipeline script (using `pytest-mock` to verify configuration handling, correct invocation of generator/writer components, and data flow).
    - **Data QA Notebook (`data/qa_analysis`):** A Jupyter Notebook for exploratory data analysis and quality assessment of the generated accelerometer data. Includes:
      - Visual inspection of time-domain signals (X, Y, Z acceleration).
      - Frequency domain analysis using FFT to validate dominant frequencies against simulation parameters.
    - **Technologies:** `pytest`, `moto`, `pytest-mock`, JupyterLab, Matplotlib, Pandas, SciPy.

6.  **Development Environment & Tooling:**
    - **Monorepo Structure:** Project organized as a monorepo with distinct Python packages.
    - **Dependency Management:** `uv` is used for Python package and environment management, configured via `pyproject.toml` files (utilizing `uv` workspaces and optional dependency groups for development tools).
    - **Version Control:** Git.

## Project Goals & Learning Objectives

This project serves as a platform to:

- Apply and deepen understanding of data engineering principles (ETL, data modeling, pipeline orchestration).
- Gain hands-on experience with modern data stack technologies (Python, Spark, Parquet, cloud services).
- Learn and implement infrastructure as code (Terraform).
- Practice robust testing methodologies (unit testing, mocking).
- Explore time-series data simulation and analysis techniques.
- Prepare for data engineering roles, particularly those involving cloud platforms and big data technologies.

## Future Development / Roadmap

The following enhancements are planned:

- **Refactor Transformation Logic:** Move data transformations (e.g., adding partition keys, type casting) into a dedicated `data/transformers` package for better modularity and reusability.
- **Implement S3 -> Timestream Loading:** Develop a mechanism (e.g., AWS Lambda, Glue ETL) to load the Parquet files from S3 into the provisioned AWS Timestream table.
- **Introduce Spark Processing:** Add a PySpark job to read data from S3, perform distributed transformations/aggregations, and write results.
- **Data Quality Framework:** Integrate Great Expectations for automated data validation within the pipeline or as part of QA.
- **Workflow Orchestration:** Implement a workflow orchestrator (e.g., Apache Airflow, Prefect, or AWS Step Functions) to manage and schedule the pipeline.
- **Enhanced Pipeline Configuration:** Make the pipeline more dynamically configurable, potentially reading pipeline definitions from a database or more complex configuration files.
- **Streaming Capabilities:** Explore a real-time version of the pipeline, potentially using Kafka/Kinesis and a streaming data generator/writer.
- **Alternative Compute:** Investigate deploying pipeline components (e.g., the `streams_service` as an API) to AWS App Runner or ECS Fargate.

## Setup & Usage (High-Level)

1.  **Prerequisites:** Python 3.12+, `uv`, Terraform, AWS CLI configured with appropriate credentials.
2.  **Environment Setup:**
    ```bash
    # From project root
    uv venv # Create virtual environment
    source .venv/bin/activate
    uv pip install -e ".[dev,test,lint,docs]" # Install all dependencies
    ```
3.  **Infrastructure Deployment:**
    ```bash
    cd infra/terraform
    terraform init
    terraform plan
    terraform apply # Review plan and confirm
    # Note the S3 bucket name from terraform output
    cd ../..
    ```
4.  **Running the Pipeline:**
    - Update the `config.yaml` (create from `config.example.yaml` if provided) with the correct S3 bucket name and other parameters.
    - Execute the pipeline script:
      ```bash
      python -m streams_service.main --config path/to/your/config.yaml
      ```
5.  **Running Tests:**
    ```bash
    # From project root
    uv run pytest
    ```
6.  **Data Analysis:**
    - Launch Jupyter Lab: `jupyter lab`
    - Navigate to `data/qa_analysis/` to open and run the analysis notebook.

## Notes

This project is primarily for learning and demonstration. While best practices are aimed for, some aspects (like the current direct transformation within the pipeline script) are slated for refactoring as the project evolves. The focus is on building a functional end-to-end system and exploring various technologies involved in data engineering.
