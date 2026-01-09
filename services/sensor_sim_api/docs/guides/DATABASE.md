# Database Management

This project uses **PostgreSQL** for storage and **Alembic** for schema migrations.

## Quick Commands

| Task         | Command           | Description                                |
| :----------- | :---------------- | :----------------------------------------- |
| **Start DB** | `make db-up`      | Start just the database container.         |
| **Connect**  | `make db-shell`   | Open a psql shell (User: `sensor_user`).   |
| **Migrate**  | `make db-migrate` | Apply all pending migrations.              |
| **Reset**    | `make db-reset`   | Destroy volumes and recreate the DB fresh. |

## Managing Migrations

**Alembic** is used to manage database schema changes. **Do not** modify the database manually using SQL; always use migrations.

### 1. Make Changes to Models

Modify the SQLAlchemy models in `services/sensor_sim_api/app/data/models/sql/`.

### 2. Generate a Migration Script

Run the following command to auto-generate a migration file based on your code changes:

```bash
make db-migrate-generate
```

_You will be prompted to enter a description (e.g., "add_gyroscope_table")._

### 3. Review the Script

Check the generated file in `services/sensor_sim_api/alembic/versions/`. Ensure the upgrade and downgrade logic looks correct.

### 4. Apply Changes

Apply the migration to your local development database:

```bash
make db-migrate
```

### 5. Verify

Check the migration status:

```bash
make db-migrate-current
```

## History & Rollbacks

- **View History:** `make db-migrate-history`
- **Undo Last Migration:** `make db-migrate-downgrade` (Rolls back 1 version)

## Database Configuration

The database connection details are defined in your environment files (`.env.dev` / `.env.test`).

- **Development Port:** 5433 (mapped to internal 5432)
- **Test Port:** 5434 (mapped to internal 5432)
  !> [!NOTE]
  > Non-standard ports are used locally to avoid conflicts with other PostgreSQL instances you might have running.
