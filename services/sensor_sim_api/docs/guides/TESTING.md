# Testing Strategy

Focus is primarily on unit code testing of app functionalities, with some very basic integration testing.

## Running Tests

Use `make` to run tests in an isolated Docker environment.

### 1. Service Tests (API)

Tests the FastAPI application, database integrations, and business logic.

```bash
make test-sensor-sim-api
```

_Time: ~5 seconds_

### 2. Generator Tests (Core Logic)

Tests the data generation algorithms (math heavy).

```bash
make test-generators
```

_Time: ~25 seconds_

### 3. Run Everything

Run the full suite across the monorepo.

```bash
make test-all
```

## Test Infrastructure

Tests run in a **completely isolated environment** defined in `docker-compose.test.yml`.

- **Database:** A separate PostgreSQL container (`sensor_sim_db_test`) running on port 5434.
- **Application:** A test runner container (`sensor_sim_api_test`) that mounts the code and runs `pytest`.

### Lifecycle

When you run `make test-sensor-sim-api`:

1. Test containers are spun up.
2. The database is reset and migrations are applied.
3. Tests run.
4. Containers are stopped and cleaned up automatically.

### Manual Control

If you need to debug the test environment:

```bash
# Start test DB and app containers
make test-db-setup

# Open a shell inside the test runner
make test-shell

# Stop and remove everything
make test-db-reset
```

## Writing Tests

### Location

- API Tests: `services/sensor_sim_api/tests/`
- Generator Tests: `data/generators/tests/`

### Principles

- **Data Isolation:** Each test should create its own data. I typically use the local `dev` environment for "dog-food" testing,
  so I didn't want the testing environment to mess with that. I will probably write some code to generate seed data down the road.
- **Pytest:** This project uses `pytest` and `pytest-asyncio` for async endpoints.

### Example Test

```python
@pytest.mark.asyncio
async def test_create_sensor(client):
    response = await client.post("/api/v1/sensors/", json={...})
    assert response.status_code == 201
    assert response.json()["id"] is not None
```
