"""Dual-plane scoring. Both planes delegate to ``biopulse-core``; no metric is re-implemented here.

  - Scientific Artifact plane: the task's opaque scorer (accuracy vs the hidden solution).
  - Agent Process plane: ``summarize_process`` over the recorded ``model_call`` / ``code_exec`` stream.

These functions only compute the verdict. Recording it onto a trace belongs to the tracer (``tracing.py``),
keeping scoring independent of the active observability backend.
"""

from __future__ import annotations

from pathlib import Path

from biopulse.runner.process_plane import summarize_process
from biopulse.tasks.registry import get as get_task

from .middleware import RunRecorder


def score_scientific(task_type: str, benchmark: Path | str, run_dir: Path | str, run_id: str) -> dict:
    """Scientific Artifact plane: dispatch to the task's opaque scorer via the task registry."""
    return get_task(task_type).scorer(benchmark, run_dir, run_id=run_id)


def score_process(recorder: RunRecorder, *, finished: bool, max_iterations: int) -> dict:
    """Agent Process plane: reduce the recorded event stream to a behavioral summary."""
    return summarize_process(
        recorder.events,
        finished=finished,
        iterations=recorder.iterations,
        max_iterations=max_iterations,
    )
