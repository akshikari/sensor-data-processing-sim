# Debugging Guide

This guide explains how to attach a debugger to the running FastAPI application inside Docker.

## Prerequisites

- A `.vscode/launch.json` comaptible IDE installed (VS Code, nvim with nvim-dap, etc.).
- **Docker** and **Docker Compose** running.

## Overview

The development container is pre-configured with `debugpy`.

- **Port 5678**: Exposed for the debugger.
- **Port 8000**: Exposed for the API.

## Configuration

To attach the debugger, you need a launch configuration.

1. Create or edit `.vscode/launch.json` in the project root (or workspace root).
2. Add the following configuration:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Docker: Attach to Sensor API",
      "type": "debugpy",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}/services/sensor_sim_api",
          "remoteRoot": "/app/services/sensor_sim_api"
        }
      ],
      "justMyCode": true
    }
  ]
}
```

**Note on Path Mappings:**
Ensure `localRoot` points to the `services/sensor_sim_api` directory on your host machine. The `remoteRoot` is fixed to `/app/services/sensor_sim_api` inside the container.

## How to Debug

1. **Start the Environment:**

   ```bash
   make dev
   ```

   Wait until you see the application startup logs.

2. **Set Breakpoints:**

3. **Attach Debugger:**

4. **Trigger Code:**
   Send a request to the API (e.g., using Swagger UI at `http://localhost:8000/docs`).
   Your IDE should pause execution at your breakpoint.

## Troubleshooting

**Debugger won't connect:**

- Check if port 5678 is actually listening: `lsof -i :5678`
- Ensure the container is running: `docker ps`
- Check logs for any startup errors: `make logs`

**Breakpoints turn gray (Unverified):**

- This usually means path mappings are incorrect.
- Verify that `${workspaceFolder}/services/sensor_sim_api` matches the actual path on your disk.
