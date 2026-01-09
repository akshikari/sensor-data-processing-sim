# Sensor Simulation API

REST API for real-time sensor data streaming using the accelerometer generator library.

## Environment Setup

### Environment Files

The project uses different `.env` files for each environment:

- `.env.dev` - Development environment (with debugger support)
- `.env.test` - Test environment (used by pytest)
- `.env.prod` - Production environment (secure, not committed to git)
- `.env.example` - Template showing all required variables

**Setup Steps:**

1. Copy `.env.example` to create your environment files:

   ```bash
   cp .env.example .env.dev
   cp .env.example .env.test
   cp .env.example .env.prod
   ```

2. Update each file with appropriate values for that environment

3. **NEVER commit `.env.dev`, `.env.test`, or `.env.prod`** - they are gitignored

## Building with Dagger

The project uses [Dagger](https://dagger.io) to build Docker images efficiently with smart caching.

From the **repository root**, run:

### Development Image (with debugger on port 5678)

```bash
make build-dev
```

Or directly:

```bash
dagger call build-dev-image --project sensor_sim_api --tag dev
```

### Production Image

```bash
make build
```

Or directly:

```bash
dagger call build-prod-image --project sensor_sim_api --tag latest
```

## Running with Docker Compose

### Development Environment

```bash
make dev
```

This will:

1. Build the dev image with Dagger
2. Start PostgreSQL database
3. Start FastAPI app with hot-reload and debugger

Access the API at: http://localhost:8000
Attach debugger at: localhost:5678

### Production Environment

```bash
make prod
```

This will:

1. Build the production image with Dagger
2. Start PostgreSQL database
3. Start FastAPI app with 4 workers (no debugger)

## Database Management

This project uses Alembic for all database schema management.

### Initial Setup

```bash
make db-up          # Start database
make db-migrate     # Run all migrations
```

### Making Schema Changes

1. Edit models in `app/data/models/sql/`
2. Generate migration:
   ```bash
   make db-migrate-generate
   # Enter descriptive message when prompted
   ```
3. Review the generated migration in `alembic/versions/`
4. Apply migration:
   ```bash
   make db-migrate
   ```

### Other Commands

```bash
make db-migrate-current      # Show current migration
make db-migrate-history      # Show all migrations
make db-migrate-downgrade    # Rollback last migration
make db-reset                # Fresh database with all migrations
make db-shell                # Open psql shell
make db-down                 # Stop database
```

**Note:** We use Alembic exclusively for schema management. All schema changes must go through Alembic migrations.

## Testing

```bash
make test          # Run tests (creates test database automatically)
```

## Other Commands

```bash
make lint          # Run ruff linter
make format        # Format code with ruff
make logs          # View service logs
make stop          # Stop all services
make clean         # Clean up containers and cache
```

## Domain Models

### Core Domains

#### Streams

## Local Development (without Docker)

```bash
uv run fastapi dev app/main.py
```
