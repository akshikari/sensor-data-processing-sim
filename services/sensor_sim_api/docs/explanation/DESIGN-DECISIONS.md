# Design Decisions

This document outlines the architectural choices made for the Sensor Simulation API.

## 1. Code Structure

3 Layers: Data (Repositories) -> Domain (Services) -> API (HTTP)

**Data Layer**: Responsible for interactions with any data stores (repositories). So far just PostgreSQL
**Domain Layer**: Responsible for parsing requests from API layer and retrieving/pushing data to the Data layer,
applying any necessary business logic or rules in between.
**API Layer**: The layer responsible for accepting requests from the client and applying basic schema validations.

### Motivation

- **Decoupling**: The core logic (Domain) knows nothing about the Database or HTTP framework.
- **Maintainability**: It's easy to determine where in the code to go for specific changes if needed.

## 2. FastAPI

### Motivation

- **Performance**: High performance (async support).
- **Type Safety**: Built on Python type hints and Pydantic.
- **Documentation**: Automatic Swagger/OpenAPI generation.

## 3. PostgreSQL & Alembic

### Motivation

- **Reliability**: Industry-standard relational database.
- **JSONB**: JSONB columns are used for `generate_data_params`. This allows for storing different configuration schemas for different sensor types without altering the table structure.
  - This decision is provisional. JSONB was selected to prioritize flexibility until schema decisions can be finalized after implementing other sensors.
- **Migrations**: Alembic provides version control for the database schema.

## 4. Dagger for CI/CD

### Motivation

- **Portability**: The build pipeline runs the same locally as it does in CI.
- **Caching**: Dagger caches intermediate build steps intelligently.
- **Language**: Pipelines are written in Python, not YAML.

!> [!NOTE]

> When developing on MacOS, the dagger calls to Docker Desktop trigger several "Allow {terminal-running-dagger} access to other apps data" prompts.
> Online resources say this is due to MacOS' security policies. This can prove to be quite a headache and the only solution found so far is to allow
> the terminal Full Disk Access. Workarounds involving writing the tarball to the `/tmp` directory and using `make` commands to make `docker` CLI
> calls to build the image from the tarball only reduced the number of security access prompts but did not get rid of the issue. For now dagger is
> implemented as intended (using the `export` function). Moving the development setup to a containerized environment may help resolve this.

## 5. Monorepo Structure

### Motivation

- **Shared Code**: The `generators` data library is shared between the API and the future Streaming Service.
- **Maintainability**: Monorepo setups have significant benefits, but can have challenges especially with projects with conflicting dependencies.
  Existing solutions for managing such a monorepo with `uv` are limited, but the current implementation has been mostly smooth so far.
- **Tooling**: Unified tooling (Make, UV, dagger) simplifies the developer workflow.
