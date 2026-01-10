"""Comprehensive tests for sensor type API endpoints."""

import asyncio
from uuid import uuid4

import pytest
from fastapi import status

pytestmark = pytest.mark.asyncio


class TestGetSensorType:
    """Tests for GET /api/v1/sensors/sensor-type/{id}"""

    async def test_get_existing_sensor_type(self, client, sensor_type):
        """Test retrieving an existing sensor type."""
        response = await client.get(f"/api/v1/sensors/sensor-type/{sensor_type.id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == str(sensor_type.id)
        assert data["name"] == sensor_type.name
        assert "create_ts" in data
        assert "update_ts" in data

    async def test_get_nonexistent_sensor_type(self, client):
        """Test retrieving a non-existent sensor type returns 404."""
        fake_id = uuid4()
        response = await client.get(f"/api/v1/sensors/sensor-type/{fake_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()

    async def test_get_with_invalid_uuid(self, client):
        """Test retrieving with invalid UUID format."""
        response = await client.get("/api/v1/sensors/sensor-type/not-a-uuid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestCreateSensorType:
    """Tests for POST /api/v1/sensors/sensor-type/"""

    async def test_create_minimal(self, client):
        """Test creation with minimal required fields."""
        payload = {"name": "gyroscope"}

        response = await client.post("/api/v1/sensors/sensor-type/", json=payload)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "id" in data
        assert data["name"] == "gyroscope"

    async def test_create_with_custom_id(self, client):
        """Test creation with a custom sensor type ID."""
        custom_id = uuid4()
        payload = {
            "id": str(custom_id),
            "name": "magnetometer",
        }

        response = await client.post("/api/v1/sensors/sensor-type/", json=payload)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["id"] == str(custom_id)
        assert data["name"] == "magnetometer"

    async def test_create_duplicate_id(self, client):
        """Test creating with duplicate ID returns 409 Conflict."""
        custom_id = uuid4()
        payload = {
            "id": str(custom_id),
            "name": "sensor1",
        }

        response1 = await client.post("/api/v1/sensors/sensor-type/", json=payload)
        assert response1.status_code == status.HTTP_201_CREATED

        response2 = await client.post("/api/v1/sensors/sensor-type/", json=payload)
        assert response2.status_code == status.HTTP_409_CONFLICT
        assert "already exists" in response2.json()["detail"].lower()

    async def test_create_missing_name(self, client):
        """Test creating without name returns 422."""
        payload = {}

        response = await client.post("/api/v1/sensors/sensor-type/", json=payload)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    async def test_create_invalid_uuid_format(self, client):
        """Test creating with invalid UUID format."""
        payload = {
            "id": "not-a-valid-uuid",
            "name": "test",
        }

        response = await client.post("/api/v1/sensors/sensor-type/", json=payload)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestUpdateSensorType:
    """Tests for PATCH /api/v1/sensors/sensor-type/{id}"""

    async def test_update_name(self, client, sensor_type):
        """Test updating sensor type name."""
        update_payload = {"name": "updated_accelerometer"}
        response = await client.patch(
            f"/api/v1/sensors/sensor-type/{sensor_type.id}", json=update_payload
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "updated_accelerometer"
        assert data["id"] == str(sensor_type.id)

    async def test_update_nonexistent_sensor_type(self, client):
        """Test updating non-existent sensor type returns 404."""
        fake_id = uuid4()
        update_payload = {"name": "doesnt_matter"}

        response = await client.patch(
            f"/api/v1/sensors/sensor-type/{fake_id}", json=update_payload
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_update_with_empty_payload(self, client, sensor_type):
        """Test update with empty payload returns 422."""
        update_payload = {}
        response = await client.patch(
            f"/api/v1/sensors/sensor-type/{sensor_type.id}", json=update_payload
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestDeleteSensorType:
    """Tests for DELETE /api/v1/sensors/sensor-type/{id}"""

    async def test_delete_existing_sensor_type(self, client):
        """Test deleting an existing sensor type."""
        # Create a sensor type first
        create_payload = {"name": "temp_sensor"}
        create_response = await client.post(
            "/api/v1/sensors/sensor-type/", json=create_payload
        )
        sensor_type_id = create_response.json()["id"]

        response = await client.delete(f"/api/v1/sensors/sensor-type/{sensor_type_id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT

        get_response = await client.get(f"/api/v1/sensors/sensor-type/{sensor_type_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

    async def test_delete_nonexistent_sensor_type(self, client):
        """Test deleting non-existent sensor type returns 404."""
        fake_id = uuid4()

        response = await client.delete(f"/api/v1/sensors/sensor-type/{fake_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_delete_already_deleted(self, client):
        """Test deleting an already deleted sensor type returns 404."""
        create_payload = {"name": "temp_sensor"}
        create_response = await client.post(
            "/api/v1/sensors/sensor-type/", json=create_payload
        )
        sensor_type_id = create_response.json()["id"]

        response1 = await client.delete(f"/api/v1/sensors/sensor-type/{sensor_type_id}")
        assert response1.status_code == status.HTTP_204_NO_CONTENT

        response2 = await client.delete(f"/api/v1/sensors/sensor-type/{sensor_type_id}")
        assert response2.status_code == status.HTTP_404_NOT_FOUND

    async def test_delete_with_invalid_uuid(self, client):
        """Test deleting with invalid UUID format."""
        response = await client.delete("/api/v1/sensors/sensor-type/not-a-uuid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestGetAllSensorTypes:
    """Tests for GET /api/v1/sensors/sensor-type/"""

    async def test_get_all_empty_list(self, client):
        """Test retrieving empty list when no sensor types exist."""
        response = await client.get("/api/v1/sensors/sensor-type/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["sensor_types"] == []
        assert data["total"] == 0

    async def test_get_all_single_sensor_type(self, client):
        """Test retrieving list with a single sensor type."""
        create_payload = {"name": "gyroscope"}
        create_response = await client.post(
            "/api/v1/sensors/sensor-type/", json=create_payload
        )
        created_id = create_response.json()["id"]

        response = await client.get("/api/v1/sensors/sensor-type/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["sensor_types"]) == 1
        assert data["total"] == 1
        assert data["sensor_types"][0]["id"] == created_id

    async def test_get_all_multiple_sensor_types(self, client):
        """Test retrieving list with multiple sensor types."""
        # Create 3 sensor types
        for i in range(3):
            payload = {"name": f"sensor_{i}"}
            response = await client.post("/api/v1/sensors/sensor-type/", json=payload)
            assert response.status_code == status.HTTP_201_CREATED

        response = await client.get("/api/v1/sensors/sensor-type/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["sensor_types"]) == 3
        assert data["total"] == 3

    async def test_get_all_ordering(self, client):
        """Test that results are ordered by create_ts DESC (newest first)."""
        # Create 3 sensor types with slight delays to ensure different timestamps
        created_ids = []
        for i in range(3):
            payload = {"name": f"sensor_{i}"}
            response = await client.post("/api/v1/sensors/sensor-type/", json=payload)
            assert response.status_code == status.HTTP_201_CREATED
            created_ids.append(response.json()["id"])
            if i < 2:
                await asyncio.sleep(0.05)

        response = await client.get("/api/v1/sensors/sensor-type/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        returned_ids = [item["id"] for item in data["sensor_types"]]

        # Newest should be first (reverse order of creation)
        assert returned_ids == list(reversed(created_ids))

    async def test_get_all_pagination_skip(self, client):
        """Test pagination with skip parameter."""
        # Create 5 sensor types
        for i in range(5):
            payload = {"name": f"sensor_{i}"}
            await client.post("/api/v1/sensors/sensor-type/", json=payload)

        response = await client.get("/api/v1/sensors/sensor-type/?skip=2")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["sensor_types"]) == 3
        assert data["total"] == 5

    async def test_get_all_pagination_limit(self, client):
        """Test pagination with limit parameter."""
        # Create 5 sensor types
        for i in range(5):
            payload = {"name": f"sensor_{i}"}
            await client.post("/api/v1/sensors/sensor-type/", json=payload)

        response = await client.get("/api/v1/sensors/sensor-type/?limit=2")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["sensor_types"]) == 2
        assert data["total"] == 5

    async def test_get_all_pagination_skip_and_limit(self, client):
        """Test pagination with both skip and limit parameters."""
        # Create 10 sensor types
        for i in range(10):
            payload = {"name": f"sensor_{i}"}
            await client.post("/api/v1/sensors/sensor-type/", json=payload)

        response = await client.get("/api/v1/sensors/sensor-type/?skip=3&limit=4")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["sensor_types"]) == 4
        assert data["total"] == 10

    async def test_get_all_excludes_archived(self, client):
        """Test that archived sensor types are excluded from results."""
        # Create 4 sensor types
        created_ids = []
        for i in range(4):
            payload = {"name": f"sensor_{i}"}
            response = await client.post("/api/v1/sensors/sensor-type/", json=payload)
            created_ids.append(response.json()["id"])

        # Archive 2 of them
        await client.delete(f"/api/v1/sensors/sensor-type/{created_ids[0]}")
        await client.delete(f"/api/v1/sensors/sensor-type/{created_ids[2]}")

        response = await client.get("/api/v1/sensors/sensor-type/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["sensor_types"]) == 2
        assert data["total"] == 2

        # Verify the archived ones are not in the list
        returned_ids = [item["id"] for item in data["sensor_types"]]
        assert created_ids[0] not in returned_ids
        assert created_ids[2] not in returned_ids
        assert created_ids[1] in returned_ids
        assert created_ids[3] in returned_ids


class TestSensorTypeDataIntegrity:
    """Integration tests for data integrity and consistency."""

    async def test_timestamps_on_create(self, client):
        """Test that create_ts and update_ts are set on creation."""
        payload = {"name": "test_sensor"}

        response = await client.post("/api/v1/sensors/sensor-type/", json=payload)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "create_ts" in data
        assert "update_ts" in data
        assert data["create_ts"] is not None
        assert data["update_ts"] is not None

    async def test_update_ts_changes_on_update(self, client):
        """Test that update_ts changes when sensor type is updated."""
        create_payload = {"name": "test_sensor"}
        create_response = await client.post(
            "/api/v1/sensors/sensor-type/", json=create_payload
        )
        sensor_type_id = create_response.json()["id"]
        original_update_ts = create_response.json()["update_ts"]

        await asyncio.sleep(0.1)

        update_payload = {"name": "updated_sensor"}
        update_response = await client.patch(
            f"/api/v1/sensors/sensor-type/{sensor_type_id}", json=update_payload
        )

        new_update_ts = update_response.json()["update_ts"]
        assert new_update_ts != original_update_ts

    async def test_create_ts_unchanged_on_update(self, client):
        """Test that create_ts doesn't change when sensor type is updated."""
        create_payload = {"name": "test_sensor"}
        create_response = await client.post(
            "/api/v1/sensors/sensor-type/", json=create_payload
        )
        sensor_type_id = create_response.json()["id"]
        original_create_ts = create_response.json()["create_ts"]

        update_payload = {"name": "updated_sensor"}
        update_response = await client.patch(
            f"/api/v1/sensors/sensor-type/{sensor_type_id}", json=update_payload
        )

        new_create_ts = update_response.json()["create_ts"]
        assert new_create_ts == original_create_ts

    async def test_full_crud_lifecycle(self, client):
        """Test complete CRUD lifecycle of a sensor type."""
        create_payload = {"name": "lifecycle_sensor"}
        create_response = await client.post(
            "/api/v1/sensors/sensor-type/", json=create_payload
        )
        assert create_response.status_code == status.HTTP_201_CREATED
        sensor_type_id = create_response.json()["id"]

        get_response = await client.get(f"/api/v1/sensors/sensor-type/{sensor_type_id}")
        assert get_response.status_code == status.HTTP_200_OK
        assert get_response.json()["id"] == sensor_type_id

        update_payload = {"name": "updated_lifecycle_sensor"}
        update_response = await client.patch(
            f"/api/v1/sensors/sensor-type/{sensor_type_id}", json=update_payload
        )
        assert update_response.status_code == status.HTTP_200_OK
        assert update_response.json()["name"] == "updated_lifecycle_sensor"

        delete_response = await client.delete(
            f"/api/v1/sensors/sensor-type/{sensor_type_id}"
        )
        assert delete_response.status_code == status.HTTP_204_NO_CONTENT

        final_get = await client.get(f"/api/v1/sensors/sensor-type/{sensor_type_id}")
        assert final_get.status_code == status.HTTP_404_NOT_FOUND
