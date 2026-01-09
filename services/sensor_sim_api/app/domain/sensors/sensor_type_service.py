"""Business logic service layer for sensor type operations."""

from uuid import UUID
from sqlalchemy.orm import Session

from app.core.exceptions import ResourceNotFoundError
from app.data.models.api_schemas import (
    SensorTypeCreate,
    SensorTypeResponse,
    SensorTypeUpdate,
)
from app.data.repositories.sensors import SensorTypeRepository
from app.data.models.sql import SensorType
from app.core.logging import get_logger

logger = get_logger(__name__)


class SensorTypeService:
    """Business logic service for sensor type operations.
    :ivar repository: Data access repository for sensor types.

    """

    def __init__(self, db: Session):
        """Initialize the service with a database session.

        :param db: SQLAlchemy session to pass to the repository for database operations.
        """
        self.repository = SensorTypeRepository(db)

    def get_sensor_type(self, id: UUID) -> SensorTypeResponse:
        """Get a sensor type by its ID

        :param id: ID of the sensor type to retrieve
        :raises ResourceNotFoundError: if the sensor type with the given ID is not
            found or has been archived.
        :return: A sensor type with the given ID if found
        """
        sensor_type = self.repository.get_sensor_type(id)
        if not sensor_type:
            raise ResourceNotFoundError("Sensor Type", str(id))
        return SensorTypeResponse.model_validate(sensor_type)

    def create_sensor_type(self, sensor_type: SensorTypeCreate) -> SensorTypeResponse:
        """Create a new sensor type.

        :param sensor_type: Validated API request schema containing sensor
            configuration data.
        :raises ResourceAlreadyExistsError: If an sensor type with the given ID already exists.
        :raises InvalidReferenceError: If creating the sensor type violates some foreign key constraint.
            Some other object needs to exist prior to creating this one.
        :raises ValidationError: If creating the sensor type violates any database schema constraints.
        :return: Newly created sensor type.
        """
        db_model = SensorType(id=sensor_type.id, name=sensor_type.name)
        created = self.repository.create_sensor_type(db_model)
        return SensorTypeResponse.model_validate(created)

    def update_sensor_type(
        self, id: UUID, updates: SensorTypeUpdate
    ) -> SensorTypeResponse:
        """Update the sensor type with the given ID. Handles partial updates of fields.

        :param id: ID of the sensor type to update
        :param updates: Object containing the fields to update and their new values.
        :raises ResourceNotFoundError: If sensor type with the given ID is not found
            or has been archived.
        :raises ValidationError: If creating the sensor type violates any database schema constraints.
        :return: Newly updated sensor type.
        """
        updates_data = updates.model_dump(exclude_unset=True)
        updated_sensor_type = self.repository.update_sensor_type(id, updates_data)
        if not updated_sensor_type:
            raise ResourceNotFoundError("Sensor Type", str(id))
        return SensorTypeResponse.model_validate(updated_sensor_type)

    def delete_sensor_type(self, id: UUID) -> None:
        """Archive an sensor type by ID

        :param id: The ID of the sensor type to be deleted.
        :raises ResourceNotFoundError: If sensor type with the given ID is not found
            or has been archived.
        :return: None
        """
        result = self.repository.delete_sensor_type(id)
        if not result:
            raise ResourceNotFoundError("Sensor Type", str(id))
        return
