# Architecture

The **Sensor Simulation API** follows **Clean Architecture** principles to separate concerns, improve testability, and ensure the business logic remains independent of frameworks and databases.

## High-Level Overview

The application is structured into concentric layers. Dependencies point inwards.

1.  **API Layer** (Outer) - Entry point (FastAPI).
2.  **Domain Layer** (Inner) - Business logic and use cases.
3.  **Data Layer** (Outer/Adapter) - Database access and persistence.

## Directory Structure

```text
app/
├── api/             # API Layer (Routes, Controllers)
│   └── sensors/     # Route modules (accelerometer.py, etc.)
├── domain/          # Domain Layer (Business Logic)
│   └── sensors/     # Service classes (accelerometer_service.py)
├── data/            # Data Layer (Persistence)
│   ├── models/      # Data definitions
│   │   ├── api_schemas/  # Pydantic models (DTOs)
│   │   └── sql/          # SQLAlchemy models (DB Tables)
│   ├── repositories/     # Data Access Objects
│   └── sources/          # Database connection setup
└── core/            # Cross-cutting concerns (Config, Logging)
```

## Layer Responsibilities

### 1. API Layer (`app/api`)

- **Role**: Handles HTTP requests and responses.
- **Responsibilities**:
  - Validating input (using Pydantic schemas).
  - Parsing parameters.
  - Calling the appropriate Domain Service.
  - Returning HTTP status codes and JSON responses.
- **Dependencies**: Depends on `Domain` and `Data` (for schemas).

### 2. Domain Layer (`app/domain`)

- **Role**: Contains the business rules and use cases.
- **Responsibilities**:
  - Orchestrating complex operations.
  - Enforcing business invariants (e.g., "Cannot change sensor type").
  - Interacting with Repositories to fetch/save data.
- **Dependencies**: Depends on `Data` (Repositories).

### 3. Data Layer (`app/data`)

- **Role**: Manages data persistence.
- **Responsibilities**:
  - **Models**: Defines database tables (`sql/`) and API transfer objects (`api_schemas/`).
  - **Repositories**: Encapsulates raw SQL/ORM queries. Provides a clean interface for the Domain layer (e.g., `get_by_id`, `create`).
  - **Sources**: Manages database connections.

## Key Design Patterns

- **Repository Pattern**: Hides the details of how data is stored. The Domain layer asks for an object, and the Repository handles the SQL.
- **Dependency Injection**: Dependencies (like the Database Session) are injected into functions, making testing easier.
- **DTOs (Data Transfer Objects)**: Pydantic models (`api_schemas`) are used to define the shape of data moving in and out of the API, separate from the internal Database Models (`sql`).
