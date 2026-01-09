from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Uuid, JSON
from sqlalchemy.orm import Mapped, mapped_column

from generators.accelerometer.models import (
    StreamState,
)

from app.data.models.sql.base import Base


class SensorType(Base):
    __tablename__: str = "sensor_types"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=lambda: uuid4())
    name: Mapped[str] = mapped_column(String)
    create_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    update_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    archived_ts: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Accelerometer(Base):
    __tablename__: str = "accelerometers"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=lambda: uuid4())
    sensor_type_id: Mapped[UUID] = mapped_column(ForeignKey("sensor_types.id"))
    generate_data_params: Mapped[dict] = mapped_column(JSON, nullable=False)
    anomalous_data_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    stream_state: Mapped[StreamState | None] = mapped_column(JSON, nullable=True)
    create_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    update_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    archived_ts: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
