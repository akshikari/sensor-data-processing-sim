"""
Module for generating mock accelerometer data
"""

from datetime import datetime

from uuid import uuid4
from pydantic import BaseModel, UUID4, FiniteFloat


class AccelerometerData(BaseModel):
    """Data model for accelerometer time series data"""

    timestamp: datetime
    sensor_id: UUID4
    accel_x: FiniteFloat
    accel_y: FiniteFloat
    accel_z: FiniteFloat


def generate_data() -> list[AccelerometerData]:
    """Generates simulated accelerometer data"""
    return [
        AccelerometerData(
            timestamp=datetime.now(),
            sensor_id=uuid4(),
            accel_x=1.2435,
            accel_y=0.58234,
            accel_z=5.4414,
        )
    ]
