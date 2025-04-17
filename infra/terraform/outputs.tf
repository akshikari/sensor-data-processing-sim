output "timestream_database_name" {
  description = "Name of the Timestream database"
  value       = aws_timestreamwrite_database.main.database_name
}

output "timestream_table_name" {
  description = "Name of the Timestream table"
  value       = aws_timestreamwrite_table.sensor_data.table_name
}

output "application_iam_role_arn" {
  description = "ARN of the IAM Role for the application"
  value       = aws_iam_role.application_role.arn
}

output "s3_data_landing_bucket_name" {
  description = "Name of the S3 for where the raw data lands"
  value       = aws_s3_bucket.raw_accelerometer_data.id
}
