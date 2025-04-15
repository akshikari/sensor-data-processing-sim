# Development Tooling

## uv

- [Docs](https://docs.astral.sh/uv/)
- General notes:
  - This [blog post](https://medium.com/@asafshakarzy/releasing-a-monorepo-using-uv-workspace-and-python-semantic-release-0dafc889f4cc)
    was a great starting point.
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
