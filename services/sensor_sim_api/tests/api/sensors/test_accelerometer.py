"""Comprehensive tests for accelerometer sensor API endpoints."""

import asyncio
from uuid import uuid4

import pytest
from fastapi import status

pytestmark = pytest.mark.asyncio


class TestGetAccelerometer:
    """Tests for GET /api/v1/sensors/accelerometer/{id}"""

    async def test_get_existing_accelerometer(self, client, sensor_type):
        """Test retrieving an existing accelerometer."""
        create_payload = {"sensor_type_id": str(sensor_type.id)}
        create_response = await client.post(
            "/api/v1/sensors/accelerometer/", json=create_payload
        )
        created_id = create_response.json()["id"]

        response = await client.get(f"/api/v1/sensors/accelerometer/{created_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == created_id
        assert data["sensor_type_id"] == str(sensor_type.id)
        assert "generate_data_params" in data
        assert "create_ts" in data
        assert "update_ts" in data

    async def test_get_nonexistent_accelerometer(self, client):
        """Test retrieving a non-existent accelerometer returns 404."""
        fake_id = uuid4()
        response = await client.get(f"/api/v1/sensors/accelerometer/{fake_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()

    async def test_get_with_invalid_uuid(self, client):
        """Test retrieving with invalid UUID format."""
        response = await client.get("/api/v1/sensors/accelerometer/not-a-uuid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestCreateAccelerometer:
    """Tests for POST /api/v1/sensors/accelerometer/"""

    async def test_create_minimal(self, client, sensor_type):
        """Test creation with minimal required fields."""
        payload = {"sensor_type_id": str(sensor_type.id)}

        response = await client.post("/api/v1/sensors/accelerometer/", json=payload)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "id" in data
        assert data["sensor_type_id"] == str(sensor_type.id)
        assert data["generate_data_params"]["gait_frequency_hz"] == 2.0
        assert data["generate_data_params"]["amplitude_sway_m"] == 0.05
        assert data["anomalous_data_params"] is None

    async def test_create_with_custom_generate_params(self, client, sensor_type):
        """Test creation with custom generation parameters."""
        payload = {
            "sensor_type_id": str(sensor_type.id),
            "generate_data_params": {
                "gait_frequency_hz": 3.0,
                "amplitude_sway_m": 0.08,
                "amplitude_bounce_m": 0.03,
                "noise_std_dev": 0.02,
            },
        }

        response = await client.post("/api/v1/sensors/accelerometer/", json=payload)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["generate_data_params"]["gait_frequency_hz"] == 3.0
        assert data["generate_data_params"]["amplitude_sway_m"] == 0.08
        assert data["generate_data_params"]["amplitude_bounce_m"] == 0.03
        assert data["generate_data_params"]["noise_std_dev"] == 0.02

    async def test_create_with_anomaly_params(self, client, sensor_type):
        """Test creation with anomalous data parameters."""
        payload = {
            "sensor_type_id": str(sensor_type.id),
            "anomalous_data_params": {
                "z_amp_modifier": 0.7,
                "step_frequency": 5,
            },
        }

        response = await client.post("/api/v1/sensors/accelerometer/", json=payload)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["anomalous_data_params"]["z_amp_modifier"] == 0.7
        assert data["anomalous_data_params"]["step_frequency"] == 5

    async def test_create_with_all_params(self, client, sensor_type):
        """Test creation with both normal and anomalous parameters."""
        payload = {
            "sensor_type_id": str(sensor_type.id),
            "generate_data_params": {
                "gait_frequency_hz": 2.5,
                "amplitude_sway_m": 0.06,
            },
            "anomalous_data_params": {
                "z_amp_modifier": 0.8,
                "step_frequency": 4,
            },
        }

        response = await client.post("/api/v1/sensors/accelerometer/", json=payload)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["generate_data_params"]["gait_frequency_hz"] == 2.5
        assert data["anomalous_data_params"]["z_amp_modifier"] == 0.8

    async def test_create_with_custom_id(self, client, sensor_type):
        """Test creation with a custom sensor ID."""
        custom_id = uuid4()
        payload = {
            "id": str(custom_id),
            "sensor_type_id": str(sensor_type.id),
        }

        response = await client.post("/api/v1/sensors/accelerometer/", json=payload)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["id"] == str(custom_id)

    async def test_create_duplicate_id(self, client, sensor_type):
        """Test creating with duplicate ID returns 409 Conflict."""
        custom_id = uuid4()
        payload = {
            "id": str(custom_id),
            "sensor_type_id": str(sensor_type.id),
        }

        response1 = await client.post("/api/v1/sensors/accelerometer/", json=payload)
        assert response1.status_code == status.HTTP_201_CREATED

        response2 = await client.post("/api/v1/sensors/accelerometer/", json=payload)
        assert response2.status_code == status.HTTP_409_CONFLICT
        assert "already exists" in response2.json()["detail"].lower()

    async def test_create_with_invalid_sensor_type(self, client):
        """Test creating with non-existent sensor_type_id returns 422."""
        fake_sensor_type_id = uuid4()
        payload = {"sensor_type_id": str(fake_sensor_type_id)}

        response = await client.post("/api/v1/sensors/accelerometer/", json=payload)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert "does not exist" in response.json()["detail"].lower()

    async def test_create_missing_sensor_type(self, client):
        """Test creating without sensor_type_id returns 422."""
        payload = {}

        response = await client.post("/api/v1/sensors/accelerometer/", json=payload)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    async def test_create_invalid_uuid_format(self, client):
        """Test creating with invalid UUID format."""
        payload = {"sensor_type_id": "not-a-valid-uuid"}

        response = await client.post("/api/v1/sensors/accelerometer/", json=payload)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    @pytest.mark.parametrize(
        "invalid_params,field_name",
        [
            ({"gait_frequency_hz": -1.0}, "gait_frequency_hz"),
            ({"gait_frequency_hz": 0.0}, "gait_frequency_hz"),
            ({"amplitude_sway_m": -0.1}, "amplitude_sway_m"),
            ({"amplitude_bounce_m": -0.1}, "amplitude_bounce_m"),
            ({"noise_std_dev": -0.01}, "noise_std_dev"),
        ],
    )
    async def test_create_invalid_generate_params(
        self, client, sensor_type, invalid_params, field_name
    ):
        """Test validation of generation parameters."""
        payload = {
            "sensor_type_id": str(sensor_type.id),
            "generate_data_params": invalid_params,
        }

        response = await client.post("/api/v1/sensors/accelerometer/", json=payload)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    @pytest.mark.parametrize(
        "invalid_params",
        [
            {"z_amp_modifier": -0.5},
            {"step_frequency": 0},
            {"step_frequency": -4},
        ],
    )
    async def test_create_invalid_anomaly_params(
        self, client, sensor_type, invalid_params
    ):
        """Test validation of anomalous data parameters."""
        payload = {
            "sensor_type_id": str(sensor_type.id),
            "anomalous_data_params": invalid_params,
        }

        response = await client.post("/api/v1/sensors/accelerometer/", json=payload)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestUpdateAccelerometer:
    """Tests for PATCH /api/v1/sensors/accelerometer/{id}"""

    async def test_update_generate_params(self, client, sensor_type):
        """Test updating generation parameters."""
        create_payload = {"sensor_type_id": str(sensor_type.id)}
        create_response = await client.post(
            "/api/v1/sensors/accelerometer/", json=create_payload
        )
        accelerometer_id = create_response.json()["id"]

        update_payload = {
            "generate_data_params": {
                "gait_frequency_hz": 3.5,
                "amplitude_sway_m": 0.1,
            }
        }
        response = await client.patch(
            f"/api/v1/sensors/accelerometer/{accelerometer_id}", json=update_payload
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["generate_data_params"]["gait_frequency_hz"] == 3.5
        assert data["generate_data_params"]["amplitude_sway_m"] == 0.1
        assert data["sensor_type_id"] == str(sensor_type.id)

    async def test_update_anomaly_params(self, client, sensor_type):
        """Test updating anomaly parameters."""
        create_payload = {
            "sensor_type_id": str(sensor_type.id),
            "anomalous_data_params": {
                "z_amp_modifier": 0.8,
                "step_frequency": 4,
            },
        }
        create_response = await client.post(
            "/api/v1/sensors/accelerometer/", json=create_payload
        )
        accelerometer_id = create_response.json()["id"]

        update_payload = {
            "anomalous_data_params": {
                "z_amp_modifier": 0.6,
                "step_frequency": 3,
            }
        }
        response = await client.patch(
            f"/api/v1/sensors/accelerometer/{accelerometer_id}", json=update_payload
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["anomalous_data_params"]["z_amp_modifier"] == 0.6
        assert data["anomalous_data_params"]["step_frequency"] == 3

    async def test_update_partial_fields(self, client, sensor_type):
        """Test partial update (PATCH semantics) - only specified fields change."""
        create_payload = {
            "sensor_type_id": str(sensor_type.id),
            "generate_data_params": {
                "gait_frequency_hz": 2.0,
                "amplitude_sway_m": 0.05,
                "noise_std_dev": 0.05,
            },
        }
        create_response = await client.post(
            "/api/v1/sensors/accelerometer/", json=create_payload
        )
        accelerometer_id = create_response.json()["id"]

        update_payload = {"generate_data_params": {"gait_frequency_hz": 2.8}}
        response = await client.patch(
            f"/api/v1/sensors/accelerometer/{accelerometer_id}", json=update_payload
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["generate_data_params"]["gait_frequency_hz"] == 2.8
        assert data["generate_data_params"]["amplitude_sway_m"] == 0.05
        assert data["generate_data_params"]["noise_std_dev"] == 0.05

    async def test_update_sensor_type_id(self, client, db_session, sensor_type):
        """Test that sensor_type_id field is ignored in updates (immutable)."""
        from app.data.models.sql import SensorType

        sensor_type_2 = SensorType(name="gyroscope")
        db_session.add(sensor_type_2)
        await db_session.commit()
        await db_session.refresh(sensor_type_2)

        create_payload = {"sensor_type_id": str(sensor_type.id)}
        create_response = await client.post(
            "/api/v1/sensors/accelerometer/", json=create_payload
        )
        accelerometer_id = create_response.json()["id"]

        update_payload = {"sensor_type_id": str(sensor_type_2.id)}
        response = await client.patch(
            f"/api/v1/sensors/accelerometer/{accelerometer_id}", json=update_payload
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # sensor_type_id should remain unchanged (field is ignored)
        assert data["sensor_type_id"] == str(sensor_type.id)

    async def test_update_nonexistent_accelerometer(self, client):
        """Test updating non-existent accelerometer returns 404."""
        fake_id = uuid4()
        update_payload = {"generate_data_params": {"gait_frequency_hz": 3.0}}

        response = await client.patch(
            f"/api/v1/sensors/accelerometer/{fake_id}", json=update_payload
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_update_with_invalid_data(self, client, sensor_type):
        """Test updating with invalid data returns 422."""
        create_payload = {"sensor_type_id": str(sensor_type.id)}
        create_response = await client.post(
            "/api/v1/sensors/accelerometer/", json=create_payload
        )
        accelerometer_id = create_response.json()["id"]

        update_payload = {"generate_data_params": {"gait_frequency_hz": -5.0}}
        response = await client.patch(
            f"/api/v1/sensors/accelerometer/{accelerometer_id}", json=update_payload
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    async def test_update_empty_payload(self, client, sensor_type):
        """Test update with empty payload (should succeed but change nothing)."""
        create_payload = {"sensor_type_id": str(sensor_type.id)}
        create_response = await client.post(
            "/api/v1/sensors/accelerometer/", json=create_payload
        )
        accelerometer_id = create_response.json()["id"]
        original_data = create_response.json()

        update_payload = {}
        response = await client.patch(
            f"/api/v1/sensors/accelerometer/{accelerometer_id}", json=update_payload
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["sensor_type_id"] == original_data["sensor_type_id"]
        assert data["generate_data_params"] == original_data["generate_data_params"]


class TestDeleteAccelerometer:
    """Tests for DELETE /api/v1/sensors/accelerometer/{id}"""

    async def test_delete_existing_accelerometer(self, client, sensor_type):
        """Test deleting an existing accelerometer."""
        create_payload = {"sensor_type_id": str(sensor_type.id)}
        create_response = await client.post(
            "/api/v1/sensors/accelerometer/", json=create_payload
        )
        accelerometer_id = create_response.json()["id"]

        response = await client.delete(
            f"/api/v1/sensors/accelerometer/{accelerometer_id}"
        )

        assert response.status_code == status.HTTP_204_NO_CONTENT

        get_response = await client.get(
            f"/api/v1/sensors/accelerometer/{accelerometer_id}"
        )
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

    async def test_delete_nonexistent_accelerometer(self, client):
        """Test deleting non-existent accelerometer returns 404."""
        fake_id = uuid4()

        response = await client.delete(f"/api/v1/sensors/accelerometer/{fake_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_delete_already_deleted(self, client, sensor_type):
        """Test deleting an already deleted accelerometer returns 404."""
        create_payload = {"sensor_type_id": str(sensor_type.id)}
        create_response = await client.post(
            "/api/v1/sensors/accelerometer/", json=create_payload
        )
        accelerometer_id = create_response.json()["id"]

        response1 = await client.delete(
            f"/api/v1/sensors/accelerometer/{accelerometer_id}"
        )
        assert response1.status_code == status.HTTP_204_NO_CONTENT

        response2 = await client.delete(
            f"/api/v1/sensors/accelerometer/{accelerometer_id}"
        )
        assert response2.status_code == status.HTTP_404_NOT_FOUND

    async def test_delete_with_invalid_uuid(self, client):
        """Test deleting with invalid UUID format."""
        response = await client.delete("/api/v1/sensors/accelerometer/not-a-uuid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestAccelerometerDataIntegrity:
    """Integration tests for data integrity and consistency."""

    async def test_timestamps_on_create(self, client, sensor_type):
        """Test that create_ts and update_ts are set on creation."""
        payload = {"sensor_type_id": str(sensor_type.id)}

        response = await client.post("/api/v1/sensors/accelerometer/", json=payload)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "create_ts" in data
        assert "update_ts" in data
        assert data["create_ts"] is not None
        assert data["update_ts"] is not None

    async def test_update_ts_changes_on_update(self, client, sensor_type):
        """Test that update_ts changes when accelerometer is updated."""
        create_payload = {"sensor_type_id": str(sensor_type.id)}
        create_response = await client.post(
            "/api/v1/sensors/accelerometer/", json=create_payload
        )
        accelerometer_id = create_response.json()["id"]
        original_update_ts = create_response.json()["update_ts"]

        await asyncio.sleep(0.1)

        update_payload = {"generate_data_params": {"gait_frequency_hz": 3.0}}
        update_response = await client.patch(
            f"/api/v1/sensors/accelerometer/{accelerometer_id}", json=update_payload
        )

        new_update_ts = update_response.json()["update_ts"]
        assert new_update_ts != original_update_ts

    async def test_create_ts_unchanged_on_update(self, client, sensor_type):
        """Test that create_ts doesn't change when accelerometer is updated."""
        create_payload = {"sensor_type_id": str(sensor_type.id)}
        create_response = await client.post(
            "/api/v1/sensors/accelerometer/", json=create_payload
        )
        accelerometer_id = create_response.json()["id"]
        original_create_ts = create_response.json()["create_ts"]

        update_payload = {"generate_data_params": {"gait_frequency_hz": 3.0}}
        update_response = await client.patch(
            f"/api/v1/sensors/accelerometer/{accelerometer_id}", json=update_payload
        )

        new_create_ts = update_response.json()["create_ts"]
        assert new_create_ts == original_create_ts

    async def test_full_crud_lifecycle(self, client, sensor_type):
        """Test complete CRUD lifecycle of an accelerometer."""
        create_payload = {
            "sensor_type_id": str(sensor_type.id),
            "generate_data_params": {"gait_frequency_hz": 2.0},
        }
        create_response = await client.post(
            "/api/v1/sensors/accelerometer/", json=create_payload
        )
        assert create_response.status_code == status.HTTP_201_CREATED
        accelerometer_id = create_response.json()["id"]

        get_response = await client.get(
            f"/api/v1/sensors/accelerometer/{accelerometer_id}"
        )
        assert get_response.status_code == status.HTTP_200_OK
        assert get_response.json()["id"] == accelerometer_id

        update_payload = {"generate_data_params": {"gait_frequency_hz": 3.0}}
        update_response = await client.patch(
            f"/api/v1/sensors/accelerometer/{accelerometer_id}", json=update_payload
        )
        assert update_response.status_code == status.HTTP_200_OK
        assert (
            update_response.json()["generate_data_params"]["gait_frequency_hz"] == 3.0
        )

        delete_response = await client.delete(
            f"/api/v1/sensors/accelerometer/{accelerometer_id}"
        )
        assert delete_response.status_code == status.HTTP_204_NO_CONTENT

        final_get = await client.get(
            f"/api/v1/sensors/accelerometer/{accelerometer_id}"
        )
        assert final_get.status_code == status.HTTP_404_NOT_FOUND


class TestGetAllAccelerometers:
    """Tests for GET /api/v1/sensors/accelerometer/"""

    async def test_get_all_empty_list(self, client):
        """Test retrieving empty list when no accelerometers exist."""
        response = await client.get("/api/v1/sensors/accelerometer/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["accelerometers"] == []
        assert data["total"] == 0

    async def test_get_all_single_accelerometer(self, client, sensor_type):
        """Test retrieving list with a single accelerometer."""
        create_payload = {"sensor_type_id": str(sensor_type.id)}
        create_response = await client.post(
            "/api/v1/sensors/accelerometer/", json=create_payload
        )
        created_id = create_response.json()["id"]

        response = await client.get("/api/v1/sensors/accelerometer/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["accelerometers"]) == 1
        assert data["total"] == 1
        assert data["accelerometers"][0]["id"] == created_id

    async def test_get_all_multiple_accelerometers(self, client, sensor_type):
        """Test retrieving list with multiple accelerometers."""
        # Create 3 accelerometers
        for _ in range(3):
            payload = {"sensor_type_id": str(sensor_type.id)}
            response = await client.post("/api/v1/sensors/accelerometer/", json=payload)
            assert response.status_code == status.HTTP_201_CREATED

        response = await client.get("/api/v1/sensors/accelerometer/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["accelerometers"]) == 3
        assert data["total"] == 3

    async def test_get_all_ordering(self, client, sensor_type):
        """Test that results are ordered by create_ts DESC (newest first)."""
        # Create 3 accelerometers with slight delays to ensure different timestamps
        created_ids = []
        for i in range(3):
            payload = {"sensor_type_id": str(sensor_type.id)}
            response = await client.post("/api/v1/sensors/accelerometer/", json=payload)
            assert response.status_code == status.HTTP_201_CREATED
            created_ids.append(response.json()["id"])
            if i < 2:
                await asyncio.sleep(0.05)

        response = await client.get("/api/v1/sensors/accelerometer/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        returned_ids = [item["id"] for item in data["accelerometers"]]

        # Newest should be first (reverse order of creation)
        assert returned_ids == list(reversed(created_ids))

    async def test_get_all_pagination_skip(self, client, sensor_type):
        """Test pagination with skip parameter."""
        # Create 5 accelerometers
        for _ in range(5):
            payload = {"sensor_type_id": str(sensor_type.id)}
            await client.post("/api/v1/sensors/accelerometer/", json=payload)

        response = await client.get("/api/v1/sensors/accelerometer/?skip=2")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["accelerometers"]) == 3
        assert data["total"] == 5

    async def test_get_all_pagination_limit(self, client, sensor_type):
        """Test pagination with limit parameter."""
        # Create 5 accelerometers
        for _ in range(5):
            payload = {"sensor_type_id": str(sensor_type.id)}
            await client.post("/api/v1/sensors/accelerometer/", json=payload)

        response = await client.get("/api/v1/sensors/accelerometer/?limit=2")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["accelerometers"]) == 2
        assert data["total"] == 5

    async def test_get_all_pagination_skip_and_limit(self, client, sensor_type):
        """Test pagination with both skip and limit parameters."""
        # Create 10 accelerometers
        for _ in range(10):
            payload = {"sensor_type_id": str(sensor_type.id)}
            await client.post("/api/v1/sensors/accelerometer/", json=payload)

        response = await client.get("/api/v1/sensors/accelerometer/?skip=3&limit=4")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["accelerometers"]) == 4
        assert data["total"] == 10

    async def test_get_all_excludes_archived(self, client, sensor_type):
        """Test that archived accelerometers are excluded from results."""
        # Create 4 accelerometers
        created_ids = []
        for _ in range(4):
            payload = {"sensor_type_id": str(sensor_type.id)}
            response = await client.post("/api/v1/sensors/accelerometer/", json=payload)
            created_ids.append(response.json()["id"])

        # Archive 2 of them
        await client.delete(f"/api/v1/sensors/accelerometer/{created_ids[0]}")
        await client.delete(f"/api/v1/sensors/accelerometer/{created_ids[2]}")

        response = await client.get("/api/v1/sensors/accelerometer/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["accelerometers"]) == 2
        assert data["total"] == 2

        # Verify the archived ones are not in the list
        returned_ids = [item["id"] for item in data["accelerometers"]]
        assert created_ids[0] not in returned_ids
        assert created_ids[2] not in returned_ids
        assert created_ids[1] in returned_ids
        assert created_ids[3] in returned_ids
