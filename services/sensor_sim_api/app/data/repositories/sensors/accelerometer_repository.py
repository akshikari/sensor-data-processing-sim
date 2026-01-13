"""Repository layer for accelerometer sensor data access."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, select, update, func
from sqlalchemy.exc import IntegrityError, OperationalError

from app.core.exceptions import (
    DatabaseError,
    InvalidReferenceError,
    ResourceAlreadyExistsError,
    ValidationError,
)
from app.data.models.sql import Accelerometer
from app.core.logging import get_logger

logger = get_logger(__name__)


class AccelerometerRepository:
    """Data access repository for accelerometer sensor CRUD operations.

    :ivar db: SQLAlchemy async database session for executing queries.
    """

    def __init__(self, db: AsyncSession):
        """Initialize the repository with an async database session.

        :param db: Active SQLAlchemy AsyncSession for database operations.
        """
        self.db = db

    async def get_all_accelerometers(
        self, skip: int = 0, limit: int = 100
    ) -> tuple[list[Accelerometer], int]:
        """Get all active accelerometers with pagination.

        :param skip: Number of records to skip (offset)
        :param limit: Maximum number of records to return
        :return: Tuple containing (list of accelerometers, total count)
        """
        # Query for items
        stmt = (
            select(Accelerometer)
            .where(Accelerometer.archived == False)
            .order_by(Accelerometer.create_ts.desc())
            .offset(skip)
            .limit(limit)
        )

        # Query for total count
        count_stmt = (
            select(func.count())
            .select_from(Accelerometer)
            .where(Accelerometer.archived == False)
        )

        try:
            result = await self.db.execute(stmt)
            items = result.scalars().all()

            count_result = await self.db.execute(count_stmt)
            total = count_result.scalar_one()

            return list(items), total
        except OperationalError as err:
            logger.error(
                f"Database operational error during get all accelerometers: {err}",
                exc_info=True,
            )
            raise DatabaseError("get all accelerometers", err) from err

    async def get_accelerometer(self, id: UUID) -> Accelerometer | None:
        """Get an accelerometer by its ID from the database.

        :param id: ID of the desired accelerometer.
        :return: The accelerometer with the given ID if it exists in the database, else None.
        """
        stmt = select(Accelerometer).where(
            and_(Accelerometer.id == id, Accelerometer.archived == False)
        )
        try:
            result = await self.db.execute(stmt)
            accelerometer = result.scalar_one_or_none()
            return accelerometer
        except OperationalError as err:
            logger.error(
                f"Database operational error during accelerometer get: {err}",
                exc_info=True,
            )
            raise DatabaseError("get accelerometer", err) from err

    async def create_accelerometer(self, accelerometer: Accelerometer) -> Accelerometer:
        """Create a new accelerometer sensor in the database.

        :param accelerometer: Accelerometer model instance to persist. Should have
            all required fields populated except auto-generated ones.
        :return: The created accelerometer instance with all fields populated,
            including auto-generated IDs and timestamps.
        :raises ResourceAlreadyExistsError: If an accelerometer with the same ID already exists
            (IntegrityError from database).
        :raises RuntimeError: If the database connection fails or other operational
            errors occur (OperationalError from database).
        """
        try:
            self.db.add(accelerometer)
            await self.db.commit()
            await self.db.refresh(accelerometer)
            return accelerometer
        except IntegrityError as err:
            error_msg = str(err.orig) if hasattr(err, "orig") else str(err)

            if "UNIQUE constraint" in error_msg or "unique" in error_msg.lower():
                raise ResourceAlreadyExistsError(
                    "Accelerometer", str(accelerometer.id)
                ) from err
            elif "foreign key constraint" in error_msg.lower():
                # TODO: Look into a better way to generate this. What if multiple foreign keys?
                raise InvalidReferenceError(
                    field="sensor_type_id",
                    value=str(accelerometer.sensor_type_id),
                    referenced_type="SensorType",
                ) from err
            else:
                raise ValidationError(
                    f"Database constraint violation: {error_msg}"
                ) from err
        except OperationalError as err:
            logger.error(
                f"Database operational error during accelerometer create: {err}",
                exc_info=True,
            )
            raise DatabaseError("create accelerometer", err) from err

    async def update_accelerometer(
        self, id: UUID, updates: dict[str, Any]
    ) -> Accelerometer | None:
        """Update an accelerometer with the provided values

        :param id: The ID of the accelerometer to update
        :param updates: Dictionary of partial or all fields to update
        :return: Updated accelerometer object
        """

        stmt = (
            update(Accelerometer)
            .where(Accelerometer.id == id)
            .values(**updates)
            .returning(Accelerometer)
        )

        try:
            result = await self.db.execute(stmt)
            await self.db.commit()
            return result.scalar_one_or_none()
        except IntegrityError as err:
            error_msg = str(err.orig) if hasattr(err, "orig") else str(err)
            raise ValidationError(
                f"Database constraint violation: {error_msg}"
            ) from err
        except OperationalError as err:
            logger.error(
                f"Database operational error during accelerometer update: {err}",
                exc_info=True,
            )
            raise DatabaseError("update accelerometer", err) from err

    async def delete_accelerometer(self, id: UUID) -> Accelerometer | None:
        """Delete an accelerometer with the provided ID

        :param id: ID of the accelerometer to delete
        :return: The deleted accelerometer or null if not found or already archived.
        """

        stmt = (
            update(Accelerometer)
            .where(
                and_(
                    Accelerometer.id == id,
                    Accelerometer.archived == False,
                )
            )
            .values({"archived": True, "archived_ts": datetime.now(timezone.utc)})
            .returning(Accelerometer)
        )

        try:
            result = await self.db.execute(stmt)
            await self.db.commit()
            return result.scalar_one_or_none()
        except OperationalError as err:
            logger.error(
                f"Database operational error during accelerometer delete: {err}",
                exc_info=True,
            )
            raise DatabaseError("delete accelerometer", err) from err
