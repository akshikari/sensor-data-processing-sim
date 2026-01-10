"""Business logic service layer for accelerometer sensor operations."""

from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ResourceNotFoundError
from app.data.models.api_schemas import (
    AccelerometerCreate,
    AccelerometerResponse,
    AccelerometerUpdate,
)
from app.data.repositories.sensors import AccelerometerRepository
from app.data.models.sql import Accelerometer
from app.core.logging import get_logger

logger = get_logger(__name__)


class AccelerometerService:
    """Business logic service for accelerometer sensor operations.
    :ivar repository: Data access repository for accelerometer sensors.

    """

    def __init__(self, db: AsyncSession):
        """Initialize the service with an async database session.

        :param db: SQLAlchemy AsyncSession to pass to the repository for database operations.
        """
        self.repository = AccelerometerRepository(db)

    async def get_accelerometer(self, id: UUID) -> AccelerometerResponse:
        """Get an accelerometer by its ID

        :param id: ID of the accelerometer to retrieve
        :return: An accelerometer with the given ID if found, else None
        """
        accelerometer = await self.repository.get_accelerometer(id)
        if not accelerometer:
            raise ResourceNotFoundError("Accelerometer", str(id))
        return AccelerometerResponse.model_validate(accelerometer)

    async def create_accelerometer(
        self, accelerometer: AccelerometerCreate
    ) -> AccelerometerResponse:
        """Create a new accelerometer sensor.

        :param accelerometer: Validated API request schema containing sensor
            configuration data.
        :return: API response schema with the created sensor data.
        :raises ResourceAlreadyExistsError: If an accelerometer with the given ID already exists.
        :raises InvalidReferenceError: If creating the accelerometer violates some foreign key constraint.
            Some other object needs to exist prior to creating this one.
        :raises ValidationError: If creating the accelerometer violates any database schema constraints.
        :return: Newly created accelerometer.
        """
        db_model = Accelerometer(
            id=accelerometer.id,
            sensor_type_id=accelerometer.sensor_type_id,
            generate_data_params=accelerometer.generate_data_params.model_dump(
                mode="json"
            ),
            anomalous_data_params=accelerometer.anomalous_data_params.model_dump(
                mode="json"
            )
            if accelerometer.anomalous_data_params
            else None,
        )
        created = await self.repository.create_accelerometer(db_model)
        return AccelerometerResponse.model_validate(created)

    async def update_accelerometer(
        self, id: UUID, updates: AccelerometerUpdate
    ) -> AccelerometerResponse:
        """Update the accelerometer with the given ID. Handles partial updates of fields.

        :param id: ID of the accelerometer to update
        :param updates: Object containing the fields to update and their new values.
        :raises ResourceNotFoundError: If the accelerometer with the given ID is not found
            or has been archived.
        :raises ValidationError: If creating the accelerometer violates any database schema constraints.
        :return: Newly updated accelerometer.
        """
        updates_data = updates.model_dump(exclude_unset=True)
        updated_accelerometer = await self.repository.update_accelerometer(
            id, updates_data
        )
        if not updated_accelerometer:
            raise ResourceNotFoundError("Accelerometer", str(id))
        return AccelerometerResponse.model_validate(updated_accelerometer)

    async def delete_accelerometer(self, id: UUID) -> None:
        """Archive an accelerometer by ID

        :param id: The ID of the accelerometer to be deleted.
        """
        result = await self.repository.delete_accelerometer(id)
        if not result:
            raise ResourceNotFoundError("Accelerometer", str(id))
        return
