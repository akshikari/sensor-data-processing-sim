resource "aws_s3_bucket" "raw_accelerometer_data" {
  # Using bucket_prefix ensures a unique name by adding a random suffix
  bucket_prefix = "${var.project_name}-raw-accelerometor-data"

  tags = {
    Project   = var.project_name
    ManagedBy = "Terraform"
    Purpose   = "Raw accelerometer sensor data landing zone"
  }
}

# Enable versioning on the bucket (good practice for data protection)
resource "aws_s3_bucket_versioning" "raw_accelerometer_data_versioning" {
  bucket = aws_s3_bucket.raw_accelerometer_data.id # Reference bucket by ID
  versioning_configuration {
    status = "Enabled"
  }
}

# Configure default server-side encryption (SSE-S3 is simplest)
resource "aws_s3_bucket_server_side_encryption_configuration" "data_landing_sse" {
  bucket = aws_s3_bucket.raw_accelerometer_data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block all public access (SECURITY BEST PRACTICE)
resource "aws_s3_bucket_public_access_block" "raw_accelerometer_data_public_access" {
  bucket = aws_s3_bucket.raw_accelerometer_data.id

  block_public_acls       = true # Block new public ACLs
  block_public_policy     = true # Block new public bucket policies
  ignore_public_acls      = true # Ignore existing public ACLs
  restrict_public_buckets = true # Restrict access if bucket becomes public via policy
}
