"""Execution backends for the agent's ``run_python`` tool.

The local backend preserves the original subprocess behavior. The Docker no-network backend is
the contamination-resistant path: it runs the same script in a container with ``--network none``,
a read-only container filesystem, and a read-only workspace mount with only ``outputs/`` writable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

from .netguard import NetPolicy, install_netguard

DEFAULT_DOCKER_IMAGE = "biopulse-runner:py312"


@dataclass(frozen=True)
class ExecutionResult:
    """Normalized subprocess result returned by every backend."""

    returncode: int
    stdout: str | bytes | None
    stderr: str | bytes | None


Runner = Callable[..., subprocess.CompletedProcess]


class LocalPythonBackend:
    """Run the generated script with the current interpreter on the host."""

    name = "local"
    network_enforced = False

    def __init__(self, runner: Runner = subprocess.run) -> None:
        self._runner = runner

    def run(
        self,
        *,
        workspace: Path,
        script_path: Path,
        timeout: int,
        net: Optional[NetPolicy] = None,
        net_log=None,
    ) -> ExecutionResult:
        env, prefix = install_netguard(net, net_log)
        completed = self._runner(
            [sys.executable, *prefix, str(script_path)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
        return ExecutionResult(completed.returncode, completed.stdout, completed.stderr)


class DockerNoNetworkBackend:
    """Run the generated script inside Docker with all container networking disabled."""

    name = "docker-none"
    network_enforced = True

    def __init__(self, *, image: str | None = None, runner: Runner = subprocess.run) -> None:
        self.image = image or os.environ.get("BIOPULSE_DOCKER_IMAGE") or DEFAULT_DOCKER_IMAGE
        self._runner = runner

    def _script_arg(self, workspace: Path, script_path: Path) -> str:
        rel = script_path.resolve().relative_to(workspace.resolve())
        return str(PurePosixPath(*rel.parts))

    def _command(self, *, workspace: Path, script_path: Path) -> list[str]:
        workspace = workspace.resolve()
        outputs = workspace / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=256m",
            "--mount",
            f"type=bind,src={workspace},dst=/workspace,readonly",
            "--mount",
            f"type=bind,src={outputs},dst=/workspace/outputs",
            "-w",
            "/workspace",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "HOME=/tmp",
        ]
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        command.extend([self.image, "python", self._script_arg(workspace, script_path)])
        return command

    def run(
        self,
        *,
        workspace: Path,
        script_path: Path,
        timeout: int,
        net: Optional[NetPolicy] = None,
        net_log=None,
    ) -> ExecutionResult:
        workspace = workspace.resolve()
        completed = self._runner(
            self._command(workspace=workspace, script_path=script_path),
            cwd=str(workspace),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return ExecutionResult(completed.returncode, completed.stdout, completed.stderr)


def make_execution_backend(name: str = "local", *, docker_image: str | None = None):
    """Return an execution backend by CLI/config name."""

    if name == "local":
        return LocalPythonBackend()
    if name == "docker-none":
        return DockerNoNetworkBackend(image=docker_image)
    raise ValueError(f"Unknown execution backend: {name}")
