# Quickstart Guide

Get the **Sensor Simulation API** running on your local machine.

## Prerequisites

- **Docker** and **Docker Compose** installed.
- **Make** (standard on Linux/macOS, use WSL/Git Bash on Windows).

## Step 1: Configuration

The project relies on environment variables. Templates are provided for easy setup.

1. Navigate to the service directory:

   ```bash
   cd services/sensor_sim_api
   ```

2. Create your local environment files:
   ```bash
   cp .env.example .env.dev    # For development
   cp .env.example .env.test   # For testing
   ```
   > **Note:** `.env.dev` and `.env.test` are gitignored. Never commit secrets.

## Step 2: Start the Environment

I recently came from a NodeJS Express project and very much enjoyed writing and using `npm` scripts.
`Makefile` commands have been great so I employ them heavily in this project.

1. Start the application in development mode:

   ```bash
   make dev
   ```

   **What happens?**
   - Docker builds the development image.
   - PostgreSQL database starts (Port 5433 to avoid conflicts).
   - FastAPI server starts with hot-reloading (Port 8000).

2. Wait for the logs to show:

   ```
   Application startup complete.
   ```

## Step 3: Verify Installation

1. Open your browser to the **API Documentation**:
   [http://localhost:8000/docs](http://localhost:8000/docs)

2. You should see the Swagger UI with available endpoints.

## Step 4: Run Your First Test

Verify that everything is working correctly by running the test suite:

```bash
make test-sensor-sim-api
```

_(This creates a separate test database container, runs tests, and cleans up automatically.)_

## Step 5: Common Commands

Keep these commands handy:

| Command         | Description                                   |
| --------------- | --------------------------------------------- |
| `make dev`      | Start the dev environment (Database + API).   |
| `make stop`     | Stop all running containers.                  |
| `make clean`    | Remove containers and build artifacts.        |
| `make logs`     | View real-time logs from the app.             |
| `make db-shell` | Open a PostgreSQL shell for the dev database. |

## Next Steps

- Check out the **[Troubleshooting Guide](../../TROUBLESHOOTING.md)** if you hit issues.
- Explore the **[Project Index](../INDEX.md)** for more documentation.
