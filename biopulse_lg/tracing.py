"""Pluggable run tracing: Langfuse (default) or LangSmith, selected by env.

Both backends expose the same three things, so the agent loop / cost ledger / scorers
stay backend-agnostic:

  1. callbacks to attach to the agent run so every model + tool call becomes a span,
  2. a trace id to score against afterwards,
  3. ``attach_scores(trace_id, scientific, process)``.

Selection (``BIOPULSE_TRACER`` wins; else auto by which creds are present; else no-op):
  - ``langfuse``  : ``LANGFUSE_PUBLIC_KEY`` + ``LANGFUSE_SECRET_KEY`` (+ ``LANGFUSE_HOST`` to self-host).
                    Traces via a CallbackHandler nested under a named root span; scores via ``create_score``.
  - ``langsmith`` : ``LANGSMITH_API_KEY`` (+ ``LANGSMITH_TRACING=true``). Traces via the env auto-tracer;
                    scores via ``create_feedback``.

``LANGSMITH_TRACING`` is forced to match the chosen backend so the LangChain auto-tracer
fires only under LangSmith, avoiding spurious 401s otherwise.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

# Scientific-plane metrics surfaced as individual scores (when present in the scorer output).
_SCI_METRIC_KEYS = ("accuracy", "schema_valid", "report_present", "macro_f1", "accuracy_scaled")


def score_items(scientific: dict, process: dict) -> Iterator[tuple[str, float, str | None]]:
    """Yield (name, value, comment) triples: the dual-plane verdict as flat scores.

    Shared by every backend so they emit an identical set of keys.
    """
    metrics = scientific.get("metrics", {}) or {}
    yield (
        "final_score",
        float(scientific.get("final_score", 0.0) or 0.0),
        f"passed={scientific.get('passed')} violations={len(scientific.get('violations', []))}",
    )
    for key in _SCI_METRIC_KEYS:
        if key in metrics:
            yield (f"sci.{key}", float(metrics[key]), None)
    yield (
        "proc.finished_cleanly",
        1.0 if process.get("finished_cleanly") else 0.0,
        f"iterations={process.get('iterations')}/{process.get('max_iterations')} "
        f"code_execs={process.get('n_code_execs')} failures={process.get('n_code_failures')}",
    )
    yield ("proc.code_error_rate", float(process.get("code_error_rate", 0.0) or 0.0), None)


@dataclass
class TraceHandle:
    """Result of ``start_run``: trace id to score later, callbacks to pass to ``invoke``."""

    trace_id: Any = None
    callbacks: list = field(default_factory=list)


class NoopTracer:
    name = "none"
    enabled = False

    @contextmanager
    def start_run(self, name: str, inputs: dict) -> Iterator[TraceHandle]:
        yield TraceHandle()

    def attach_scores(self, trace_id: Any, *, scientific: dict, process: dict) -> bool:
        return False

    def url(self, trace_id: Any) -> str | None:
        return None


class LangfuseTracer:
    name = "langfuse"
    enabled = True

    def __init__(self) -> None:
        from langfuse import get_client

        self._client = get_client()  # reads LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST from env

    @contextmanager
    def start_run(self, name: str, inputs: dict) -> Iterator[TraceHandle]:
        # Tracing must never fail the run: setup errors degrade to an untraced handle.
        try:
            from langfuse.langchain import CallbackHandler

            # Named root observation sets the trace name + inputs; the callback handler nests
            # every model/tool span under it via the active OTel context.
            observation = self._client.start_as_current_observation(name=name, as_type="span", input=inputs)
        except Exception as exc:
            print(f"[langfuse] trace setup failed ({exc}) — running untraced")
            yield TraceHandle()
            return
        with observation:
            try:
                handle = TraceHandle(trace_id=self._client.get_current_trace_id(), callbacks=[CallbackHandler()])
            except Exception as exc:
                print(f"[langfuse] handler setup failed ({exc}) — running untraced")
                handle = TraceHandle()
            try:
                yield handle
            finally:
                try:
                    self._client.flush()
                except Exception:
                    pass

    def attach_scores(self, trace_id: Any, *, scientific: dict, process: dict) -> bool:
        if not trace_id:
            return False
        try:
            for name, value, comment in score_items(scientific, process):
                self._client.create_score(
                    trace_id=trace_id, name=name, value=value, comment=comment, data_type="NUMERIC"
                )
            self._client.flush()
            return True
        except Exception as exc:  # tracing must never fail the run
            print(f"[langfuse] score attach skipped: {exc}")
            return False

    def url(self, trace_id: Any) -> str | None:
        try:
            return self._client.get_trace_url(trace_id=trace_id)
        except Exception:
            return None


class LangSmithTracer:
    name = "langsmith"
    enabled = True

    @contextmanager
    def start_run(self, name: str, inputs: dict) -> Iterator[TraceHandle]:
        # Tracing must never fail the run: setup errors degrade to an untraced handle.
        try:
            from langsmith import trace

            # The env auto-tracer captures the spans; we need only the root id (== trace_id) to score.
            with trace(name=name, inputs=inputs, project_name=os.environ.get("LANGSMITH_PROJECT")) as root:
                yield TraceHandle(trace_id=getattr(root, "id", None), callbacks=[])
                return
        except Exception as exc:
            print(f"[langsmith] trace setup failed ({exc}) — running untraced")
        yield TraceHandle()

    def attach_scores(self, trace_id: Any, *, scientific: dict, process: dict) -> bool:
        if not trace_id:
            return False
        try:
            from langsmith import Client

            client = Client()
            for name, value, comment in score_items(scientific, process):
                client.create_feedback(
                    run_id=trace_id, trace_id=trace_id, key=name, score=value, comment=comment
                )
            return True
        except Exception as exc:
            print(f"[langsmith] feedback attach skipped: {exc}")
            return False

    def url(self, trace_id: Any) -> str | None:
        return None


def select_tracer():
    """Return the env-selected tracer and align ``LANGSMITH_TRACING`` so the auto-tracer
    fires only when LangSmith is the chosen backend."""
    choice = os.environ.get("BIOPULSE_TRACER", "").strip().lower()
    has_langfuse = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
    has_langsmith = bool(os.environ.get("LANGSMITH_API_KEY"))
    if not choice:
        choice = "langfuse" if has_langfuse else "langsmith" if has_langsmith else "none"

    tracer: Any = NoopTracer()
    if choice == "langfuse":
        if not has_langfuse:
            print("[tracing] langfuse selected but LANGFUSE_PUBLIC_KEY/SECRET_KEY unset — tracing off")
        else:
            try:
                tracer = LangfuseTracer()
            except Exception as exc:
                print(f"[tracing] langfuse init failed ({exc}) — tracing off")
    elif choice == "langsmith":
        if not has_langsmith:
            print("[tracing] langsmith selected but LANGSMITH_API_KEY unset — tracing off")
        else:
            tracer = LangSmithTracer()

    # The LangChain auto-tracer keys off LANGSMITH_TRACING; enable it only for the LangSmith backend.
    os.environ["LANGSMITH_TRACING"] = "true" if isinstance(tracer, LangSmithTracer) else "false"
    return tracer
