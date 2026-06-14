"""Offline tests for the pluggable tracer seam — no Langfuse/LangSmith creds or network needed."""

from __future__ import annotations

import pytest

from biopulse_lg.tracing import NoopTracer, score_items, select_tracer

_TRACER_ENV = (
    "BIOPULSE_TRACER",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
)


@pytest.fixture
def clean_env(monkeypatch):
    for var in _TRACER_ENV:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_no_creds_selects_noop(clean_env):
    tracer = select_tracer()
    assert tracer.name == "none" and tracer.enabled is False
    # Noop is a usable context manager that yields an empty handle and never attaches.
    with tracer.start_run("biopulse_run", {"run_id": "x"}) as run:
        assert run.trace_id is None and run.callbacks == []
    assert tracer.attach_scores(None, scientific={}, process={}) is False


def test_langsmith_selected_when_keyed(clean_env):
    clean_env.setenv("LANGSMITH_API_KEY", "ls-fake")
    clean_env.setenv("BIOPULSE_TRACER", "langsmith")
    import os

    tracer = select_tracer()
    assert tracer.name == "langsmith"
    assert os.environ["LANGSMITH_TRACING"] == "true"  # auto-tracer armed only for LangSmith


def test_langfuse_without_keys_falls_back_and_silences_langsmith(clean_env):
    clean_env.setenv("BIOPULSE_TRACER", "langfuse")  # requested, but no LANGFUSE_* keys present
    import os

    tracer = select_tracer()
    assert tracer.name == "none"  # gracefully off, not a crash
    assert os.environ["LANGSMITH_TRACING"] == "false"  # LangChain auto-tracer kept quiet


def test_score_items_flattens_both_planes():
    scientific = {
        "final_score": 0.44,
        "passed": True,
        "violations": [],
        "metrics": {"accuracy": 0.44, "schema_valid": 1.0, "report_present": 1.0},
    }
    process = {
        "finished_cleanly": True,
        "code_error_rate": 0.25,
        "iterations": 4,
        "max_iterations": 20,
        "n_code_execs": 4,
        "n_code_failures": 1,
    }
    items = list(score_items(scientific, process))
    names = {name for name, _, _ in items}
    assert {"final_score", "sci.accuracy", "sci.schema_valid", "proc.finished_cleanly", "proc.code_error_rate"} <= names
    assert all(isinstance(value, float) for _, value, _ in items)
    by_name = {name: value for name, value, _ in items}
    assert by_name["final_score"] == 0.44
    assert by_name["proc.finished_cleanly"] == 1.0
