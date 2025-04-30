# Development Tooling

## uv

- [Docs](https://docs.astral.sh/uv/)
- General notes:
  - This [blog post](https://medium.com/@asafshakarzy/releasing-a-monorepo-using-uv-workspace-and-python-semantic-release-0dafc889f4cc) was a great starting point.
  - Essentially the best way I found to use `uv` to manage a "monorepo" is as follows:
    - Initial setup (starting from root-level directory):
      - `uv init --name sensor-data-sim` to set up the root-level pyproject.toml
      - `mkdir ./data/sim_data_gen && cd ./data/sim_data_gen && uv init --lib sim-data-gen` to set up nested library
        - if you want a package instead just do `uv init --package sim-data-gen` at the end instead
        - it should automatically add the nested package as a 'member' of the root-level project
    - For adding dependencies (often to specific packages within the repo):
      - `uv add --group dev scipy-stubs` for adding dev dependencies
      - `uv add scipy` if within the specific packages directory where you have already initialized the package
        - otherwise `uv add scipy --package sim-data-gen`
    - `uv lock` once you've added all the necessary dependencies
    - To setup/initialize the Python virtual environment
      - `uv venv`
      - `source ./.venv/bin/activate`
    - The quick and dirty command to make sure all your dependencies are installed: `uv sync --all-packages --all-groups`
      - Running `uv sync` within a package directory will only install the dependencies specified there

## terraform

- Great IaC tool for provisioning infrastructure
- Can track changes between the existing state and any updates made to the Terraform scripts
- Stitched together some examples from online to come up with the current format:
  - `providers.tf`: Set up AWS as the infra provider with "us-east-2" as the default region
  - `s3.tf`: Set up S3 buckets and access and security policies
  - `timestream.tf`: Set up Timestream DB (InfluxDB), tables and access and security polcies.
  - `iam.tf`: Set up IAM roles and policies
  - `outputs.tf`: Set up variables that record the output of other resources to be used as needed in setting up the rest
  - `variables.tf`: A place to store common variables that can be referenced by any other Terraform resources
- `terraform plan`: Command to see the outline of the expected output of your current Terraform scripts.
- `terraform apply`: Execute the Terraform scripts to create the desired end state of your resources
