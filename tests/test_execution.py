"""Offline tests for run_python execution backends."""

from __future__ import annotations

from types import SimpleNamespace

from biopulse_lg import tools
from biopulse_lg.execution import DockerNoNetworkBackend, ExecutionResult
from biopulse_lg.middleware import RunRecorder


def test_docker_none_backend_builds_no_network_command(tmp_path):
    workspace = tmp_path / "workspace"
    scripts = workspace / ".scripts"
    scripts.mkdir(parents=True)
    script = scripts / "step_01.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    captured = {}

    def fake_runner(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    backend = DockerNoNetworkBackend(image="biopulse-test:py312", runner=fake_runner)
    result = backend.run(workspace=workspace, script_path=script, timeout=17)

    cmd = captured["cmd"]
    assert result.returncode == 0 and result.stdout == "ok\n"
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert cmd[cmd.index("--network") + 1] == "none"
    assert "--read-only" in cmd
    assert cmd[cmd.index("--security-opt") + 1] == "no-new-privileges"
    mounts = [cmd[i + 1] for i, token in enumerate(cmd) if token == "--mount"]
    assert f"type=bind,src={workspace.resolve()},dst=/workspace,readonly" in mounts
    assert f"type=bind,src={(workspace / 'outputs').resolve()},dst=/workspace/outputs" in mounts
    assert cmd[-3:] == ["biopulse-test:py312", "python", ".scripts/step_01.py"]
    assert captured["kwargs"]["cwd"] == str(workspace.resolve())
    assert captured["kwargs"]["timeout"] == 17


def test_execute_python_records_enforced_backend(tmp_path, monkeypatch):
    class FakeBackend:
        name = "docker-none"
        network_enforced = True

        def run(self, **kwargs):
            return ExecutionResult(returncode=0, stdout="done\n", stderr="")

    monkeypatch.setattr(tools, "make_execution_backend", lambda name, docker_image=None: FakeBackend())
    recorder = RunRecorder()
    output = tools.execute_python(
        tmp_path,
        recorder,
        "import requests\nprint('done')\n",
        execution_backend="docker-none",
        docker_image="biopulse-test:py312",
    )

    assert "[exit code 0]" in output
    details = recorder.events[0]["details"]
    assert details["execution_backend"] == "docker-none"
    assert details["network_enforced"] is True
    assert details["used_network"] is True
