from typing import (
    Annotated,
    TypeAlias,
)

import dagger
import tomli
from dagger import (
    BuildArg,
    Container,
    DefaultPath,
    File,
    Ignore,
    dag,
    function,
    object_type,
)

IGNORE = Ignore(
    [
        ".env",
        ".git",
        "**/.venv",
        "**__pycache__**",
        ".dagger/sdk",
        "**/.pytest_cache",
        "**/.ruff_cache",
    ]
)

# this represents the repo root
RootDir: TypeAlias = Annotated[
    dagger.Directory,
    DefaultPath("."),
    IGNORE,
]


@object_type
class CiDagger:
    @function
    async def build_project(
        self,
        root_dir: RootDir,
        project: str,
        include_dev_deps: bool = True,
    ) -> Container:
        """Build a container containing only the source code for a given project and it's dependencies."""
        # we start by creating a container including only third-party dependencies
        # with no source code (except pyproject.toml and uv.lock from the repo root)
        container = self._container_with_third_party_dependencies(
            pyproject_toml=root_dir.file("pyproject.toml"),
            uv_lock=root_dir.file("uv.lock"),
            dockerfile=root_dir.file("Dockerfile.dev"),
            project=project,
            include_dev_deps=include_dev_deps,
        )

        # find the source code locations for the dependencies of a given project
        project_sources_map = await self._get_project_sources_map(
            root_dir.file("uv.lock"), project
        )

        container = self._copy_source_code(container, root_dir, project_sources_map)

        # fix permissions for non-root user
        container = container.with_exec(
            ["chown", "-R", "appuser:appgroup", "/app"], use_entrypoint=True
        )

        # we run `uv sync` to create editable installs of the local dependencies
        # pointing (for now) to the dummy directories we created in the previous step
        container = self._install_local_dependencies(
            container, project, include_dev_deps
        )

        # change the working directory to the project's source directory
        # so that commands in CI are automatically run in the context of this project
        container = container.with_workdir(f"/app/{project_sources_map[project]}")

        return container

    def _container_with_third_party_dependencies(
        self,
        pyproject_toml: File,
        uv_lock: File,
        dockerfile: File,
        project: str,
        include_dev_deps: bool = True,
    ) -> Container:
        # create an empty directory to make sure only the pyproject.toml
        # and uv.lock files are copied to the build context (to affect caching)
        build_context = (
            dag.directory()
            .with_file(
                "pyproject.toml",
                pyproject_toml,
            )
            .with_file(
                "uv.lock",
                uv_lock,
            )
            .with_file(
                "/Dockerfile",
                dockerfile,
            )
            .with_new_file("README.md", "Dummy README.md")
        )

        target = "deps-dev" if include_dev_deps else "deps-prod"
        return build_context.docker_build(
            target=target,
            dockerfile="/Dockerfile",
            build_args=[BuildArg(name="PACKAGE", value=project)],
        )

    async def _get_project_sources_map(
        self,
        uv_lock: File,
        project: str,
    ) -> dict[str, str]:
        """Returns a dictionary of the local dependencies' (of a given project) source directories."""
        uv_lock_dict = tomli.loads(await uv_lock.contents())
        members = set(uv_lock_dict["manifest"]["members"])

        # first, find the dependencies of our project
        local_projects = {project}

        def find_deps_for_package(package_name: str):
            for package in uv_lock_dict["package"]:
                if package["name"] == package_name:
                    dependencies = package.get("dependencies", [])
                    for dep in dependencies:
                        if isinstance(dep, dict) and dep.get("name") in members:
                            local_projects.add(dep["name"])
                            find_deps_for_package(dep["name"])

        find_deps_for_package(project)

        # now, gather all the directories with the dependency sources
        project_sources_map = {}
        for package in uv_lock_dict["package"]:
            if package["name"] in local_projects:
                project_sources_map[package["name"]] = package["source"]["editable"]

        return project_sources_map

    @function
    async def get_project_version(self, root_dir: RootDir, project: str) -> str:
        """Get the version string for a project from its pyproject.toml.

        Example:
            dagger call get-project-version --project sensor-sim-api
        """
        uv_lock_file = root_dir.file("uv.lock")
        uv_lock_dict = tomli.loads(await uv_lock_file.contents())

        # Find project path
        project_path = None
        for package in uv_lock_dict["package"]:
            if package["name"] == project:
                project_path = package["source"]["editable"]
                break

        if not project_path:
            raise ValueError(f"Project '{project}' not found in uv.lock")

        # Read version
        pyproject_file = root_dir.file(f"{project_path}/pyproject.toml")
        pyproject_dict = tomli.loads(await pyproject_file.contents())
        return pyproject_dict["project"]["version"]

    def _copy_source_code(
        self,
        container: Container,
        root_dir: RootDir,
        project_sources_map: dict[str, str],
    ) -> Container:
        for project, project_source_path in project_sources_map.items():
            container = container.with_directory(
                f"/app/{project_source_path}",
                root_dir.directory(project_source_path),
            )

        return container

    def _install_local_dependencies(
        self, container: Container, project: str, include_dev_deps: bool = True
    ) -> Container:
        cmd = ["uv", "sync", "--inexact", "--package", project]
        if not include_dev_deps:
            cmd.append("--no-dev")

        container = container.with_exec(cmd)

        return container

    @function
    async def pytest(self, root_dir: RootDir, project: str) -> str:
        """Run pytest for a given project."""
        container = await self.build_project(root_dir, project)
        return await container.with_exec(["pytest"]).stdout()

    @function
    async def pyright(self, root_dir: RootDir, project: str) -> str:
        """Run pyright for a given project."""
        container = await self.build_project(root_dir, project)
        return await container.with_exec(["pyright"]).stdout()

    @function
    async def build_dev_image(
        self,
        root_dir: RootDir,
        project: str,
    ) -> Container:
        """Build a development Docker image container with debugger support.

        Returns a Container that can be used with export-image to load into Docker.
        The container includes exposed ports for the app (8000) and debugger (5678),
        and has a 'version' label containing the project version.
        """
        version = await self.get_project_version(root_dir, project)
        container = await self.build_project(root_dir, project, include_dev_deps=True)
        container = container.with_exposed_port(5678)  # debugger port
        container = container.with_exposed_port(8000)  # app port
        container = container.with_label("version", version)

        return container

    @function
    async def build_prod_image(
        self,
        root_dir: RootDir,
        project: str,
    ) -> Container:
        """Build a production Docker image container.

        Returns a Container configured for production use.
        The container has a 'version' label containing the project version.

        Note: The command to run the application should be specified in docker-compose
        or when running the container (e.g., fastapi run --workers 4 app/main.py).
        """
        version = await self.get_project_version(root_dir, project)
        container = await self.build_project(root_dir, project, include_dev_deps=False)
        container = container.with_exposed_port(8000)
        container = container.with_label("version", version)

        return container
