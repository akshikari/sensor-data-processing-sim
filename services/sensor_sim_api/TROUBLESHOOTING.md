# Troubleshooting Guide

Solutions to common issues encountered when working with the Sensor Simulation API.

## Database Issues

### "Connection Refused" or "Cannot connect to server"
**Symptoms:** The API fails to start, logs show `Is the server running on host "db" ...?`
**Solutions:**
1. Ensure the database container is running:
   ```bash
   docker ps
   ```
2. If the container is missing, restart the environment:
   ```bash
   make stop
   make dev
   ```
3. Check if the database is healthy:
   ```bash
   docker-compose logs db
   ```

### "Port is already allocated"
**Symptoms:** Docker fails to start with error `Bind for 0.0.0.0:5433 failed: port is already allocated`.
**Cause:** Another service (or an old container) is using the project's ports.
**Solution:**
- **Port 5433:** Used by the Dev Database. Check if another Postgres instance is running locally or stop old containers:
  ```bash
  make stop
  # If that doesn't work, find the process:
  lsof -i :5433
  ```

### Migration Errors ("Target database is not up to date")
**Symptoms:** API errors related to missing tables or columns.
**Solution:**
Apply the latest migrations:
```bash
make db-migrate
```

## Docker & Dagger Issues

### Slow Builds
**Cause:** Dagger caches builds, but sometimes the cache needs refreshing or network is slow.
**Solution:**
- Dagger usually handles this well. If you suspect a bad state, you can try cleaning:
  ```bash
  make clean
  make dev
  ```

### "Container name already in use"
**Symptoms:** `Error response from daemon: Conflict. The container name "/sensor_sim_api_app" is already in use`.
**Solution:**
Cleanup orphaned containers:
```bash
make clean
```

## Testing Issues

### Tests hanging or failing to connect to DB
**Cause:** The test environment might be stuck or misconfigured.
**Solution:**
1. Force cleanup of test containers:
   ```bash
   make test-db-down
   ```
2. Run tests again:
   ```bash
   make test-sensor-sim-api
   ```

## Still Stuck?

1. **Check Logs:** `make logs` gives you the output from the API.
2. **Reset Everything:**
   ```bash
   make clean
   make dev
   ```
