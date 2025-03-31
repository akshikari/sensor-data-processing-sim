from sim_data_gen.accelerometer import AccelerometerData, generate_data


def test_generate_data():
    """Test to ensure the accelerometer data is valid"""
    for data in generate_data():
        assert type(data) == AccelerometerData
