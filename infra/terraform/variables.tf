variable "aws_region" {
  description = "AWS region to deploy resources in"
  type        = string
  default     = "us-east-2"
}

variable "project_name" {
  description = "Base name for project resources"
  type        = string
  default     = "sensor-sim"
}

variable "timestream_memory_retention_hours" {
  description = "Timestream memory store retention period in hours (min 1, max 8766)"
  type        = number
  default     = 24 # Keep recent data in memory for 1 day
}

variable "timestream_magnetic_retention_days" {
  description = "Timestream magnetic store retention period in days (min 1, max 73000)"
  type        = number
  default     = 7 # Keep older data in magnetic store for 7 days
}
