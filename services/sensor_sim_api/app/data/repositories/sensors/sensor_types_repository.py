"""Repository layer for sensor type data access."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID
from sqlalchemy import and_, select, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.core.exceptions import (
    DatabaseError,
    ResourceAlreadyExistsError,
    ValidationError,
)
from app.data.models.sql import SensorType
from app.core.logging import get_logger

logger = get_logger(__name__)


class SensorTypeRepository:
    """Data access repository for sensor type CRUD operations.

    :ivar db: SQLAlchemy database session for executing queries.
    """

    def __init__(self, db: Session):
        """Initialize the repository with a database session.

        :param db: Active SQLAlchemy session for database operations.
        """
        self.db = db

    def get_sensor_type(self, id: UUID) -> SensorType | None:
        """Get an sensor type by its ID from the database.

        :param id: ID of the desired sensor type.
        :return: The sensor type with the given ID if it exists in the database, else None.
        """
        stmt = select(SensorType).where(
            and_(SensorType.id == id, SensorType.archived == False)
        )
        try:
            result = self.db.execute(stmt)
            sensor_type = result.scalar_one_or_none()
            return sensor_type
        except OperationalError as err:
            logger.error(
                f"Database operational error during sensor type get: {err}",
                exc_info=True,
            )
            raise DatabaseError("get sensor type", err) from err

    def create_sensor_type(self, sensor_type: SensorType) -> SensorType:
        """Create a new sensor type in the database.

        :param sensor_type: SensorType model instance to persist. Should have
            all required fields populated except auto-generated ones.
        :return: The created sensor type instance with all fields populated,
            including auto-generated IDs and timestamps.
        :raises ResourceAlreadyExistsError: If an sensor type with the same ID already exists
            (IntegrityError from database).
        :raises RuntimeError: If the database connection fails or other operational
            errors occur (OperationalError from database).
        """
        try:
            self.db.add(sensor_type)
            self.db.commit()
            self.db.refresh(sensor_type)
            return sensor_type
        except IntegrityError as err:
            error_msg = str(err.orig) if hasattr(err, "orig") else str(err)

            if "UNIQUE constraint" in error_msg or "unique" in error_msg.lower():
                raise ResourceAlreadyExistsError(
                    "Sensor Type", str(sensor_type.id)
                ) from err
            else:
                raise ValidationError(
                    f"Database constraint violation: {error_msg}"
                ) from err
        except OperationalError as err:
            logger.error(
                f"Database operational error during sensor type create: {err}",
                exc_info=True,
            )
            raise DatabaseError("create sensor type", err) from err

    def update_sensor_type(
        self, id: UUID, updates: dict[str, Any]
    ) -> SensorType | None:
        """Update an sensor type with the provided values

        :param id: The ID of the sensor type to update
        :param updates: Dictionary of partial or all fields to update
        :return: Updated sensor type object
        """

        stmt = (
            update(SensorType)
            .where(SensorType.id == id)
            .values(**updates)
            .returning(SensorType)
        )

        try:
            result = self.db.execute(stmt)
            self.db.commit()
            return result.scalar_one_or_none()
        except IntegrityError as err:
            error_msg = str(err.orig) if hasattr(err, "orig") else str(err)
            raise ValidationError(
                f"Database constraint violation: {error_msg}"
            ) from err
        except OperationalError as err:
            logger.error(
                f"Database operational error during sensor type update: {err}",
                exc_info=True,
            )
            raise DatabaseError("update sensor type", err) from err

    def delete_sensor_type(self, id: UUID) -> SensorType | None:
        """Delete an sensor type with the provided ID

        :param id: ID of the sensor type to delete
        :return:
        """

        stmt = (
            update(SensorType)
            .where(
                and_(
                    SensorType.id == id,
                    SensorType.archived == False,
                )
            )
            .values({"archived": True, "archived_ts": datetime.now(timezone.utc)})
            .returning(SensorType)
        )

        try:
            result = self.db.execute(stmt)
            self.db.commit()
            return result.scalar_one_or_none()
        except OperationalError as err:
            logger.error(
                f"Database operational error during sensor type delete: {err}",
                exc_info=True,
            )
            raise DatabaseError("delete sensor type", err) from err
