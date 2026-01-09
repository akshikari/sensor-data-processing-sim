# Sensor Simulation API

**A high-performance REST API for simulating robot IMU sensor data streams.**

[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.124+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](Dockerfile)

## Overview

The **Sensor Simulation API** is the core control plane for the Sensor Data Processing Simulator. It allows users to:

- Manage sensor configurations ([Accelerometers](docs/GLOSSARY.md#accelerometer), [Gyroscopes](docs/GLOSSARY.md#gyroscope)).
- Simulate realistic sensor noise, drift, and bias.
- Stream generated data to downstream processing pipelines.

Built with **FastAPI** and **PostgreSQL**, following **Clean Architecture** principles to ensure scalability and maintainability.

## Quick Links

| Resource                                                   | Description                      |
| ---------------------------------------------------------- | -------------------------------- |
| **[Quickstart Guide](docs/getting-started/QUICKSTART.md)** | Get up and running in 5 minutes. |
| **[Documentation Index](docs/INDEX.md)**                   | Full documentation hub.          |
| **[Troubleshooting](TROUBLESHOOTING.md)**                  | Solutions for common issues.     |
| **[Glossary](docs/GLOSSARY.md)**                           | Domain terminology.              |

## Key Features

- **Realistic Simulation**: Uses statistical models to generate noisy sensor data.
- **RESTful Management**: logical endpoints for sensor CRUD operations.
- **Robust Storage**: PostgreSQL with SQLAlchemy ORM and Alembic migrations.
- **Developer Experience**:
  - Hot-reloading in development.
  - VS Code Debugger integration within Docker.
  - Comprehensive test suite (Service + Data layers).

## Minimal Setup

```bash
# 1. Configure environment
cp .env.example .env.dev

# 2. Start development environment (requires Docker)
make dev
```

Visit **[http://localhost:8000/docs](http://localhost:8000/docs)** to explore the API interactively.

## Testing

```bash
make test-sensor-sim-api   # Run API service tests
make test-all              # Run all tests in the monorepo
```

## Project Structure

- `app/api` - Routes and Request/Response handling.
- `app/domain` - Business logic and use cases.
- `app/data` - Database models and repositories.
- `app/core` - Configuration and infrastructure code.
