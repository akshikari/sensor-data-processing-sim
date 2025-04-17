# IAM Policy allowing writing to the specific Timestream table
resource "aws_iam_policy" "timestream_write_policy" {
  name        = "${var.project_name}-timestream-write-policy"
  description = "Allow writing records to the Timestream sensor data table"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "timestream:WriteRecords"
        ]
        Effect   = "Allow"
        Resource = aws_timestreamwrite_table.sensor_data.arn
      },
      {
        Action = [
          "timestream:DescribeEndpoints"
        ]
        Effect   = "Allow"
        Resource = "*" # DescribeEndpoints requires "*" resource
      }
    ]
  })
}

resource "aws_iam_role" "application_role" {
  name = "${var.project_name}-app-role"
  description = "Role assumed by the sensor data processing application/service"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = [
            "ec2.amazonaws.com",
            "lambda.amazonaws.com",
            "ecs-tasks.amazonaws.com",
            "apprunner.amazonaws.com"
          ]
        }
      }
    ]
  })

  tags = {
    Project   = var.project_name
    ManagedBy = "Terraform"
  }
}

# Attach the write policy to the role
resource "aws_iam_role_policy_attachment" "timestream_write_attach" {
  role       = aws_iam_role.application_role.name
  policy_arn = aws_iam_policy.timestream_write_policy.arn
}

# IAM Policy allowing writing to the specific S3 bucket
resource "aws_iam_policy" "s3_landing_write_policy" {
  name        = "${var.project_name}-s3-landing-write-policy"
  description = "Allow writing objects to the data landing S3 bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:PutObject"
          # Add "s3:GetObject", "s3:ListBucket" later if your code needs them
        ]
        Effect   = "Allow"
        Resource = [
          "${aws_s3_bucket.raw_accelerometer_data.arn}/*" # Access to objects WITHIN the bucket
        ]
      },
      {
        "Action": "s3:ListBucket",
        "Effect": "Allow",
        "Resource": aws_s3_bucket.raw_accelerometer_data.arn # Access TO the bucket itself
      }
    ]
  })
}

# Attach the new S3 policy to the existing application role
resource "aws_iam_role_policy_attachment" "s3_landing_write_attach" {
  role       = aws_iam_role.application_role.name
  policy_arn = aws_iam_policy.s3_landing_write_policy.arn
}
