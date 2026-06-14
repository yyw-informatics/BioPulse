"""Middleware for the ``create_agent`` loop, plus the run's event recorder.

Two middleware components define the harness-specific control flow:

  ``FinishOnOutputs`` — ``after_model`` returns ``{"jump_to": "end"}`` once the task's
      required output files exist on disk. The jump is only wired because the hook declares
      ``@hook_config(can_jump_to=["end"])``; without it the routing dict is ignored. ``jump_to``
      is ephemeral, so the check runs every iteration in ``after_model``.

  ``ProcessEvents`` — ``after_model`` logs one ``model_call`` event per turn and enforces the
      step cap directly. The cap is enforced here rather than via ``recursion_limit`` because
      that limit defaults to 10007 and exceeding it raises ``GraphRecursionError`` (a crash,
      not a clean stop).

``RunRecorder`` is the shared sink: ``run_python`` appends ``code_exec`` events, these
middlewares append ``model_call`` events, and ``run.py`` hands ``recorder.events`` to
``biopulse.runner.process_plane.summarize_process``. The keys ``event_type``, ``ok``,
``timed_out``, ``has_code``, ``is_finish`` are what that reducer reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, hook_config


class RunRecorder:
    """Mutable, single-run event log shared by the tool and the middleware."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.iterations = 0  # model calls this run; the loop-cap counter
        self.finished = False  # True once FinishOnOutputs fires the end-jump
        self._script_count = 0

    def next_script_index(self) -> int:
        self._script_count += 1
        return self._script_count

    def record_model_call(self, *, has_code: bool, is_finish: bool) -> None:
        self.iterations += 1
        self.events.append(
            {"event_type": "model_call", "details": {"has_code": has_code, "is_finish": is_finish}}
        )

    def record_code_exec(self, *, ok: bool, timed_out: bool, **details) -> None:
        # The core reducer reads ok/timed_out; harness-specific details feed behavior_summary.json.
        self.events.append(
            {"event_type": "code_exec", "details": {"ok": ok, "timed_out": timed_out, **details}}
        )


def _messages(state: Any) -> list:
    if isinstance(state, dict):
        return state.get("messages", []) or []
    return getattr(state, "messages", []) or []


class FinishOnOutputs(AgentMiddleware):
    """End the run as soon as every required output file exists in the workspace."""

    def __init__(self, workspace: Path, required_outputs: list[str], recorder: RunRecorder) -> None:
        super().__init__()
        self.workspace = Path(workspace)
        self.required_outputs = list(required_outputs)
        self.recorder = recorder

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: Any, runtime: Any) -> dict | None:  # noqa: ANN401
        if all((self.workspace / rel).exists() for rel in self.required_outputs):
            self.recorder.finished = True
            return {"jump_to": "end"}
        return None


class ProcessEvents(AgentMiddleware):
    """Log a ``model_call`` event per turn and enforce the step cap (jump to END at the cap)."""

    def __init__(self, recorder: RunRecorder, *, max_iterations: int) -> None:
        super().__init__()
        self.recorder = recorder
        self.max_iterations = max_iterations

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: Any, runtime: Any) -> dict | None:  # noqa: ANN401
        messages = _messages(state)
        last = messages[-1] if messages else None
        tool_calls = getattr(last, "tool_calls", None) or []
        has_code = any((tc.get("name") if isinstance(tc, dict) else None) == "run_python" for tc in tool_calls)
        # A turn with no tool call is the agent's natural stop; treat it as the "finish" signal
        # for the process plane's n_no_code_turns accounting.
        is_finish = len(tool_calls) == 0
        self.recorder.record_model_call(has_code=has_code, is_finish=is_finish)

        if self.recorder.iterations >= self.max_iterations and not self.recorder.finished:
            return {"jump_to": "end"}
        return None
