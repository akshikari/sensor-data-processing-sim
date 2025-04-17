# Timestream Database
resource "aws_timestreamwrite_database" "main" {
  database_name = "${var.project_name}-db"

  tags = {
    Project   = var.project_name
    ManagedBy = "Terraform"
  }
}

# Timestream Table
resource "aws_timestreamwrite_table" "sensor_data" {
  database_name = aws_timestreamwrite_database.main.database_name
  table_name    = "accelerometer-data" # Specific table name

  retention_properties {
    memory_store_retention_period_in_hours  = var.timestream_memory_retention_hours
    magnetic_store_retention_period_in_days = var.timestream_magnetic_retention_days
  }

  magnetic_store_write_properties {
    enable_magnetic_store_writes = true
  }

  tags = {
    Project   = var.project_name
    ManagedBy = "Terraform"
  }
}
