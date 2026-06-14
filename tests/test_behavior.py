"""Offline tests for the behavior summary and the per-exec capture it consumes."""

from __future__ import annotations

from biopulse_lg.behavior import summarize_behavior
from biopulse_lg.middleware import RunRecorder
from biopulse_lg.tools import execute_python


def _exec_event(ok: bool, **details):
    return {"event_type": "code_exec", "details": {"ok": ok, "timed_out": False, **details}}


def test_summarize_behavior_synthetic():
    events = [
        {"event_type": "model_call", "details": {"has_code": True, "is_finish": False}},
        _exec_event(False, duration_s=1.0, error_type="ImportError", n_lines=10, used_subprocess=True, used_network=False, wrote_output=False),
        _exec_event(False, duration_s=0.5, error_type="KeyError", n_lines=12, used_subprocess=False, used_network=True, wrote_output=False),
        _exec_event(True, duration_s=2.0, error_type=None, n_lines=20, used_subprocess=False, used_network=False, wrote_output=True),
    ]
    b = summarize_behavior(events)
    assert b["n_code_execs"] == 3 and b["n_code_failures"] == 2
    assert b["error_types"] == {"ImportError": 1, "KeyError": 1}
    assert b["recovered"] is True  # last exec succeeded
    assert b["n_recovered_failures"] == 2  # each failure was followed by a success
    assert b["n_subprocess_escapes"] == 1 and b["n_network_intent_execs"] == 1
    assert b["turns_to_first_output"] == 3 and b["n_exploration_execs"] == 2
    assert b["total_exec_seconds"] == 3.5 and b["mean_code_lines"] == 14.0


def test_behavior_tolerates_unenriched_events():
    # Pre-enrichment events carry only ok/timed_out; summary must not crash.
    b = summarize_behavior([_exec_event(True), _exec_event(False)])
    assert b["n_code_execs"] == 2 and b["error_types"] == {} and b["turns_to_first_output"] is None


def test_execute_python_enriches_events(tmp_path):
    recorder = RunRecorder()
    execute_python(tmp_path, recorder, "import os\nprint('hello')\n")
    d = [e for e in recorder.events if e["event_type"] == "code_exec"][0]["details"]
    assert d["ok"] is True and d["used_subprocess"] is False and d["n_lines"] >= 2 and "duration_s" in d


def test_execute_python_flags_subprocess_and_error(tmp_path):
    recorder = RunRecorder()
    execute_python(tmp_path, recorder, "import subprocess\nraise KeyError('boom')\n")
    d = [e for e in recorder.events if e["event_type"] == "code_exec"][0]["details"]
    assert d["ok"] is False and d["used_subprocess"] is True and d["error_type"] == "KeyError"
