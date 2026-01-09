# Development Setup

This guide will help you set up a fully functional development environment for the **Sensor Simulation API**.

## Prerequisites

Ensure you have the following installed:

- **Docker Desktop** (Engine 20.10+)
- **Make** (GNU Make 4.0+)
- **Git**
- **Dagger** (Neat CI/CD pipeline tool)
- **uv** (Fast Python package installer) - _Optional, for local tooling_

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sensor-data-processing-sim.git
cd sensor-data-processing-sim
```

## 2. Environment Configuration

The project uses environment variables for configuration. A template `.env.example` has been provided.

Navigate to the service directory and copy the examples:

```bash
cd services/sensor_sim_api
cp .env.example .env.dev
cp .env.example .env.test
```

> **Security Note:** Never commit `.env` files to version control. They are ignored by git.

## 3. Build & Start

```bash
# From the project root directory
make dev
```

This command will:

1. Build the Docker image using **Dagger** (or standard Docker build).
2. Start PostgreSQL container on port **5433**.
3. Start the FastAPI service container on port **8000**.
4. Enable **Hot Reloading** for code changes.
5. Enable **Debug Mode** (Port 5678).

## 4. Verify Setup

1. **Check Logs:**

   ```bash
   make logs
   ```

   You should see `Application startup complete`.

2. **Access API:**
   Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

3. **Check Database:**

   ```bash
   make db-shell
   ```

   This should open a `psql` prompt. Type `\q` to exit.

## 5. Local Python Setup (Optional)

If you want IntelliSense and linting in your IDE (like VS Code), you should create a local virtual environment.

**uv** is used for dependency management.

```bash
# Install dependencies
uv sync
```

I got lazy about running `uv sync` within each project, so I just run ` uv sync --all-packages --all-groups --all-extras`

## Common Tasks

| Task               | Command                     |
| :----------------- | :-------------------------- |
| **Stop Server**    | `make stop`                 |
| **Restart Server** | `make stop` then `make dev` |
| **Run Tests**      | `make test-all`             |
| **Lint Code**      | `make lint`                 |
| **Format Code**    | `make format`               |
