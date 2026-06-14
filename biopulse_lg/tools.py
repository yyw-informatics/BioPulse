"""The ``run_python`` tool — the agent's only way to act.

Writes the model's code to a per-step file under the run workspace and runs it as a subprocess
whose ``cwd`` is the workspace, so relative paths resolve naturally (read ``input/train.h5ad``,
write ``outputs/prediction.h5ad``). The default local backend inherits this process's interpreter
and environment, so ``anndata`` / ``numpy`` / ``scikit-learn`` (from ``biopulse-core``) are importable.

``execute_python`` is the framework-agnostic core shared by the LangChain ``StructuredTool`` and the
OpenAI Agents SDK ``function_tool``.

For contamination-resistant runs, select the ``docker-none`` execution backend. It runs the script in
Docker with ``--network none`` and only the run workspace mounted.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from langchain_core.tools import StructuredTool

from .execution import make_execution_backend
from .middleware import RunRecorder
from .netguard import NetPolicy

_DEFAULT_TIMEOUT = 300  # seconds
_STREAM_CAP = 6000  # chars per stream returned to the model (head+tail kept)

#: Tool description shown to the model. Shared verbatim by both framework wrappers.
RUN_PYTHON_DESCRIPTION = """Execute Python in the task workspace and return its exit status, stdout, and stderr.

The code runs with the workspace as the current directory. Read the task inputs from `input/` and WRITE
your deliverables under `outputs/` (e.g. outputs/prediction.h5ad and outputs/report.md). numpy, anndata,
and scikit-learn are importable. Each call is a FRESH process with no memory of previous calls — persist
everything you need to files, not to in-memory variables."""


def _clip(text) -> str:
    if isinstance(text, bytes):  # TimeoutExpired can carry undecoded bytes even with encoding set
        text = text.decode("utf-8", "replace")
    text = text or ""
    if len(text) <= _STREAM_CAP:
        return text
    half = _STREAM_CAP // 2
    return f"{text[:half]}\n...[{len(text) - _STREAM_CAP} chars truncated]...\n{text[-half:]}"


def _render(header: str, stdout: str | None, stderr: str | None) -> str:
    return f"{header}\n--- stdout ---\n{_clip(stdout)}\n--- stderr ---\n{_clip(stderr)}"


# Behavioral signals parsed from each execution; feed biopulse_lg.behavior, never the core reducer.
_SUBPROCESS_RE = re.compile(r"\bsubprocess\b|os\.system|os\.popen|\bPopen\b|check_call|check_output|pip install|!pip")
_NETWORK_RE = re.compile(
    r"\bimport\s+requests\b|\bimport\s+urllib\b|\bimport\s+httpx\b|from\s+urllib|requests\.(get|post|put)|"
    r"\bhttpx\.|urlopen|urllib\.request|socket\.(socket|create_connection|getaddrinfo)"
)
_ERROR_RE = re.compile(r"\b([A-Za-z_]+(?:Error|Exception))\b")


def _analyze_code(code: str) -> dict:
    return {
        "n_lines": code.count("\n") + 1,
        "used_subprocess": bool(_SUBPROCESS_RE.search(code)),  # escapes the socket netguard
        "used_network": bool(_NETWORK_RE.search(code)),  # in-process network intent, gated by netguard
        "wrote_output": "outputs/" in code,  # attempt to write deliverables vs pure exploration
    }


def _error_type(stderr: str | bytes | None) -> str:
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "replace")
    matches = _ERROR_RE.findall(stderr or "")
    return matches[-1] if matches else "Unknown"


def execute_python(
    workspace: Path,
    recorder: RunRecorder,
    code: str,
    *,
    timeout: int = _DEFAULT_TIMEOUT,
    net: "NetPolicy | None" = None,
    net_log=None,
    execution_backend: str = "local",
    docker_image: str | None = None,
) -> str:
    """Run ``code`` as a subprocess in the workspace, record a ``code_exec`` event, and return its exit
    status plus stdout/stderr. Framework-agnostic core behind both agent loops' ``run_python`` tool.

    ``net`` (a :class:`NetPolicy`) runs the local child under the Python socket-layer netguard for
    harness levels L1/L2+. ``execution_backend='docker-none'`` additionally enforces no network at
    the container boundary.
    """
    # Resolve to an ABSOLUTE path: both the subprocess cwd and the script path must be absolute. If both
    # are relative, the child resolves the relative script path against its own relative cwd and the path
    # doubles (e.g. workspace/<workspace>/.scripts/step.py), so nothing runs.
    workspace = Path(workspace).resolve()
    scripts_dir = workspace / ".scripts"  # outside outputs/ so the scorer's safety scan ignores it
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / f"step_{recorder.next_script_index():02d}.py"
    script_path.write_text(code, encoding="utf-8")
    analysis = _analyze_code(code)
    executor = make_execution_backend(execution_backend, docker_image=docker_image)
    analysis.update(
        {
            "execution_backend": executor.name,
            "network_enforced": executor.network_enforced,
            "network_policy": getattr(net, "mode", None),
        }
    )
    started = time.perf_counter()
    try:
        completed = executor.run(
            workspace=workspace, script_path=script_path, timeout=timeout, net=net, net_log=net_log
        )
        ok = completed.returncode == 0
        recorder.record_code_exec(
            ok=ok, timed_out=False, duration_s=round(time.perf_counter() - started, 3),
            error_type=(None if ok else _error_type(completed.stderr)), **analysis,
        )
        return _render(f"[exit code {completed.returncode}]", completed.stdout, completed.stderr)
    except subprocess.TimeoutExpired as exc:
        recorder.record_code_exec(
            ok=False, timed_out=True, duration_s=round(time.perf_counter() - started, 3),
            error_type="Timeout", **analysis,
        )
        return _render(f"[timed out after {timeout}s]", exc.stdout, exc.stderr)
    except Exception as exc:  # spawn/OS/decode errors: feed the agent the error, don't kill the run
        recorder.record_code_exec(
            ok=False, timed_out=False, duration_s=round(time.perf_counter() - started, 3),
            error_type=type(exc).__name__, **analysis,
        )
        return _render(f"[execution error: {type(exc).__name__}: {exc}]", "", "")


def make_run_python_tool(
    workspace: Path, recorder: RunRecorder, *, timeout: int = _DEFAULT_TIMEOUT,
    net: "NetPolicy | None" = None, net_log=None, execution_backend: str = "local",
    docker_image: str | None = None,
) -> StructuredTool:
    """Build the LangChain ``run_python`` tool bound to one run's workspace and event recorder. Binding
    the workspace here (not as a tool arg) keeps the working directory out of the model's reach. ``net``
    is the harness-level network policy applied to each execution (None = inherited network)."""

    def run_python(code: str) -> str:
        return execute_python(
            workspace,
            recorder,
            code,
            timeout=timeout,
            net=net,
            net_log=net_log,
            execution_backend=execution_backend,
            docker_image=docker_image,
        )

    run_python.__doc__ = RUN_PYTHON_DESCRIPTION
    return StructuredTool.from_function(run_python)
