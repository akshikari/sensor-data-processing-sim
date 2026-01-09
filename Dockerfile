# Build stage
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install build dependencies for psycopg2
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Copy application code
COPY ./pyproject.toml ./uv.lock ./alembic.ini /app/
COPY ./app /app/app
COPY ./scripts /app/scripts

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

# Runtime stage
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install runtime dependencies (only libpq5 for PostgreSQL client)
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code from builder
COPY --from=builder /app/app /app/app
COPY --from=builder /app/scripts /app/scripts
COPY --from=builder /app/pyproject.toml /app/uv.lock /app/alembic.ini /app/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Run application
CMD ["fastapi", "run", "--workers", "4", "app/main.py"]
