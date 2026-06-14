"""Extended Process-plane behavior metrics for biopulse-langgraph runs.

Complements biopulse-core's framework-agnostic ``summarize_process`` with harness-specific
per-execution detail: error-type breakdown, recovery, exploration
before committing, code volume, timing, and subprocess escapes that bypass the socket
netguard. Reads the same ``recorder.events`` and never alters
the core summary.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def summarize_behavior(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce recorded events to extended behavior metrics.

    Tolerant of pre-enrichment events: missing detail keys simply don't contribute.
    """
    details = [e["details"] for e in events if e.get("event_type") == "code_exec"]
    n = len(details)
    failures = [d for d in details if not d.get("ok")]
    durations = [d["duration_s"] for d in details if d.get("duration_s") is not None]
    lines = [d["n_lines"] for d in details if d.get("n_lines") is not None]
    error_types = Counter(d.get("error_type") for d in failures if d.get("error_type"))

    # A failed exec followed by any later successful one.
    recovered_failures = sum(
        1 for i, d in enumerate(details) if not d.get("ok") and any(later.get("ok") for later in details[i + 1 :])
    )
    # 1-based index of the first exec that tried to write outputs.
    first_write = next((i + 1 for i, d in enumerate(details) if d.get("wrote_output")), None)

    return {
        "schema_version": "biopulse.behavior_summary.v1",
        "n_code_execs": n,
        "n_code_failures": len(failures),
        "code_error_rate": round(len(failures) / n, 4) if n else 0.0,
        "error_types": dict(error_types),  # e.g. {"ImportError": 2, "KeyError": 1}
        "recovered": bool(details[-1].get("ok")) if details else False,  # ended on a successful exec
        "n_recovered_failures": recovered_failures,
        "total_exec_seconds": round(sum(durations), 2),
        "mean_exec_seconds": round(sum(durations) / len(durations), 2) if durations else 0.0,
        "mean_code_lines": round(sum(lines) / len(lines), 1) if lines else 0.0,
        "turns_to_first_output": first_write,  # exploration length before first write
        "n_exploration_execs": sum(1 for d in details if not d.get("wrote_output")),
        "n_subprocess_escapes": sum(1 for d in details if d.get("used_subprocess")),  # bypasses the netguard
        "n_network_intent_execs": sum(1 for d in details if d.get("used_network")),
    }
