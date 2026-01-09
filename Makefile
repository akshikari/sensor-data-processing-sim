.PHONY: help dev prod stop test-all test-services test-sensor-sim-api test-data test-generators test-db-setup test-db-down test-db-reset test-db-shell test-shell test-logs db-up db-down db-shell db-reset db-migrate db-migrate-generate db-migrate-downgrade db-migrate-history db-migrate-current clean build build-dev logs lint format

PROJECT = sensor-sim-api
PROJECT_DIR = sensor_sim_api
ENV_FILE = services/$(PROJECT_DIR)/.env.dev
POSTGRES_USER = $(shell grep POSTGRES_USER $(ENV_FILE) | cut -d '=' -f2)
POSTGRES_DB = $(shell grep POSTGRES_DB $(ENV_FILE) | cut -d '=' -f2)
DB_CONTAINER = sensor_sim_db
TEST_ENV_FILE = services/$(PROJECT_DIR)/.env.test
TEST_DB_CONTAINER = sensor_sim_db_test
TEST_APP_CONTAINER = sensor_sim_api_test
TEST_POSTGRES_USER = $(shell grep POSTGRES_USER $(TEST_ENV_FILE) | cut -d '=' -f2)
TEST_POSTGRES_DB = $(shell grep POSTGRES_DB $(TEST_ENV_FILE) | cut -d '=' -f2)

help:
	@echo "Available commands:"
	@echo ""
	@echo "=== Building ==="
	@echo "  make build-dev     - Build development Docker image (with debugger)"
	@echo "  make build         - Build production Docker image"
	@echo ""
	@echo "=== Development ==="
	@echo "  make dev           - Start development environment (app + database)"
	@echo "  make prod          - Start production environment"
	@echo "  make stop          - Stop all services"
	@echo "  make logs          - Show service logs"
	@echo ""
	@echo "=== Database ==="
	@echo "  make db-up                  - Start database only"
	@echo "  make db-down                - Stop database only"
	@echo "  make db-shell               - Open PostgreSQL shell"
	@echo "  make db-reset               - Reset database (drop, recreate, migrate)"
	@echo "  make db-migrate             - Run pending Alembic migrations"
	@echo "  make db-migrate-generate    - Generate new migration from model changes"
	@echo "  make db-migrate-downgrade   - Rollback last migration"
	@echo "  make db-migrate-history     - Show migration history"
	@echo "  make db-migrate-current     - Show current migration"
	@echo ""
	@echo "=== Testing ==="
	@echo ""
	@echo "Run All Tests:"
	@echo "  make test-all                - Everything (services + data, ~30s)"
	@echo ""
	@echo "Services:"
	@echo "  make test-services           - All service tests (~5s)"
	@echo "  make test-sensor-sim-api     - sensor-sim-api only (~5s)"
	@echo ""
	@echo "Data Libraries:"
	@echo "  make test-data               - All data libraries (~25s)"
	@echo "  make test-generators         - generators only (~25s)"
	@echo ""
	@echo "=== Test Infrastructure ==="
	@echo "  make test-db-setup           - Setup test database (runs migrations, then stops)"
	@echo "  make test-db-down            - Stop test containers"
	@echo "  make test-db-reset           - Reset test database (remove volumes)"
	@echo "  make test-db-shell           - Open PostgreSQL shell for test database"
	@echo "  make test-shell              - Open shell in test container"
	@echo "  make test-logs               - Show test container logs"
	@echo ""
	@echo "=== Utilities ==="
	@echo "  make clean         - Remove containers, volumes, and cache"
	@echo "  make lint          - Run linting checks"
	@echo "  make format        - Format code"

build-dev:
	@echo "🔍 Getting project version..."
	@VERSION=$$(dagger call get-project-version --project $(PROJECT)); \
	echo "🔨 Building and exporting dev image (version: $$VERSION)..."; \
	dagger call build-dev-image --project $(PROJECT) export-image --name $(PROJECT):$$VERSION-dev; \
	docker tag $(PROJECT):$$VERSION-dev $(PROJECT):dev; \
	echo "✅ Built $(PROJECT):$$VERSION-dev (also tagged as :dev)"

build:
	@echo "🔍 Getting project version..."
	@VERSION=$$(dagger call get-project-version --project $(PROJECT)); \
	echo "🔨 Building and exporting prod image (version: $$VERSION)..."; \
	dagger call build-prod-image --project $(PROJECT) export-image --name $(PROJECT):$$VERSION; \
	docker tag $(PROJECT):$$VERSION $(PROJECT):latest; \
	echo "✅ Built $(PROJECT):$$VERSION (also tagged as :latest)"

dev: build-dev
	@echo "Starting development environment..."
	docker compose up

prod: build
	@echo "Starting production environment..."
	docker compose -f docker-compose.prod.yml up -d

stop:
	docker compose down
	docker compose -f docker-compose.prod.yml down

test-sensor-sim-api:
	@echo "Running sensor-sim-api tests..."
	@docker compose -f docker-compose.test.yml up -d db-test app-test || \
		(echo "❌ Failed to start test containers"; docker compose -f docker-compose.test.yml down; exit 1)
	@echo "Waiting for containers to be ready..."
	@sleep 2
	@echo ""
	@echo "==================== RUNNING TESTS ===================="
	@echo ""
	@docker compose -f docker-compose.test.yml logs -f app-test & \
		LOGS_PID=$$!; \
		docker wait $(TEST_APP_CONTAINER) > /dev/null 2>&1; \
		EXIT_CODE=$$(docker inspect $(TEST_APP_CONTAINER) --format='{{.State.ExitCode}}'); \
		kill $$LOGS_PID 2>/dev/null || true; \
		echo ""; \
		echo "==================== CLEANING UP ====================="; \
		echo ""; \
		docker compose -f docker-compose.test.yml down || \
			(echo "❌ Failed to stop containers - possible hanging process!"; exit 1); \
		if [ $$EXIT_CODE -eq 0 ]; then \
			echo "✓ sensor-sim-api tests passed (35 tests)"; \
		else \
			echo "❌ sensor-sim-api tests failed (exit code: $$EXIT_CODE)"; \
		fi; \
		exit $$EXIT_CODE

test-services: test-sensor-sim-api
	@echo ""
	@echo "=========================================="
	@echo "✓ ALL SERVICE TESTS PASSED"
	@echo "  - sensor-sim-api: 35 tests"
	@echo "=========================================="

test-generators:
	@echo "Running generator tests..."
	@cd data/generators && uv run pytest tests/ -q || \
		(echo ""; echo "❌ Generator tests failed"; \
		 echo ""; echo "Re-running failed tests with verbose output..."; \
		 echo ""; uv run pytest tests/ --lf -v; exit 1)
	@echo "✓ Generator tests passed"

test-data: test-generators
	@echo ""
	@echo "=========================================="
	@echo "✓ ALL DATA LIBRARY TESTS PASSED"
	@echo "  - generators: tests passed"
	@echo "=========================================="

test-all: test-services test-data
	@echo ""
	@echo "=========================================="
	@echo "✓ ALL TESTS PASSED"
	@echo ""
	@echo "Services:"
	@echo "  - sensor-sim-api: 35 tests (~5s)"
	@echo ""
	@echo "Data Libraries:"
	@echo "  - generators: 53 tests (~25s)"
	@echo ""
	@echo "Total: ~30s"
	@echo "=========================================="

test-db-setup:
	@echo "Setting up test database..."
	@docker compose -f docker-compose.test.yml up -d db-test || \
		(echo "❌ Failed to start test database"; exit 1)
	@echo "Waiting for test database to be ready..."
	@MAX_ATTEMPTS=30; \
	ATTEMPT=0; \
	until docker exec $(TEST_DB_CONTAINER) pg_isready -U $(TEST_POSTGRES_USER) -d $(TEST_POSTGRES_DB) > /dev/null 2>&1; do \
		ATTEMPT=$$((ATTEMPT + 1)); \
		if [ $$ATTEMPT -ge $$MAX_ATTEMPTS ]; then \
			echo "❌ Test database failed to start after $$MAX_ATTEMPTS attempts"; \
			docker compose -f docker-compose.test.yml down; \
			exit 1; \
		fi; \
		echo "  Waiting for PostgreSQL... (attempt $$ATTEMPT/$$MAX_ATTEMPTS)"; \
		sleep 2; \
	done
	@echo "Running migrations on test database..."
	@docker compose -f docker-compose.test.yml run --rm app-test \
		sh -c "cd /app/services/sensor_sim_api && alembic upgrade head" || \
		(echo "❌ Migrations failed"; docker compose -f docker-compose.test.yml down; exit 1)
	@echo "Stopping test containers..."
	@docker compose -f docker-compose.test.yml down || \
		(echo "❌ Failed to stop containers - possible hanging process!"; exit 1)
	@echo "✓ Test database setup complete"

test-db-down:
	@echo "Stopping test containers..."
	@docker compose -f docker-compose.test.yml down

test-db-reset:
	@echo "Resetting test environment (removing volumes)..."
	@docker compose -f docker-compose.test.yml down -v
	@echo "✓ Test environment reset complete"
	@echo ""
	@echo "Run 'make test-db-setup' to reinitialize"

test-db-shell:
	@echo "Opening PostgreSQL shell for test database..."
	@docker compose -f docker-compose.test.yml up -d db-test
	@echo "Waiting for database to be ready..."
	@until docker exec $(TEST_DB_CONTAINER) pg_isready -U $(TEST_POSTGRES_USER) -d $(TEST_POSTGRES_DB) > /dev/null 2>&1; do \
		sleep 1; \
	done
	@docker exec -it $(TEST_DB_CONTAINER) psql -U $(TEST_POSTGRES_USER) -d $(TEST_POSTGRES_DB)
	@echo ""
	@echo "Shell closed. Run 'make test-db-down' to stop the database."

test-shell:
	@echo "Opening shell in test container..."
	@docker compose -f docker-compose.test.yml up -d db-test
	@echo "Waiting for database to be ready..."
	@until docker exec $(TEST_DB_CONTAINER) pg_isready -U $(TEST_POSTGRES_USER) -d $(TEST_POSTGRES_DB) > /dev/null 2>&1; do \
		sleep 1; \
	done
	@docker compose -f docker-compose.test.yml run --rm app-test sh
	@echo ""
	@echo "Shell closed. Run 'make test-db-down' to stop the database."

test-logs:
	@docker compose -f docker-compose.test.yml logs -f

db-up:
	docker compose up db -d
	@echo "Waiting for database to be ready..."
	@until docker exec $(DB_CONTAINER) pg_isready -U $(POSTGRES_USER) > /dev/null 2>&1; do sleep 1; done
	@echo "✓ Database ready"

db-down:
	docker compose stop db

db-shell:
	docker exec -it $(DB_CONTAINER) psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

db-reset:
	@echo "Resetting database..."
	docker compose down -v
	docker compose up db -d
	@echo "Waiting for database to be ready..."
	@until docker exec $(DB_CONTAINER) pg_isready -U $(POSTGRES_USER) > /dev/null 2>&1; do sleep 1; done
	@echo "✓ Database ready, running migrations..."
	$(MAKE) db-migrate

db-migrate:
	@echo "Running database migrations..."
	cd services/$(PROJECT_DIR) && uv run alembic upgrade head

db-migrate-generate:
	@echo "Generating new migration..."
	@read -p "Enter migration message: " msg; \
	cd services/$(PROJECT_DIR) && uv run alembic revision --autogenerate -m "$$msg"

db-migrate-downgrade:
	@echo "Rolling back last migration..."
	cd services/$(PROJECT_DIR) && uv run alembic downgrade -1

db-migrate-history:
	@echo "Migration history:"
	cd services/$(PROJECT_DIR) && uv run alembic history --verbose

db-migrate-current:
	@echo "Current migration:"
	cd services/$(PROJECT_DIR) && uv run alembic current

logs:
	docker compose logs -f

clean:
	docker compose down -v
	docker system prune -f
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

lint:
	uv run ruff check .

format:
	uv run ruff format .
