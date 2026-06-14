"""OpenAI Agents SDK loop — a parallel engine to the LangChain ``create_agent`` path.

The evaluation core (workspace, ``run_python`` tool, scorers, cost contract, tracing) is
framework-agnostic; only the loop differs here:

  - ``Agent(instructions=…, tools=[run_python], model=…)`` + ``Runner.run_sync(max_turns=…)``.
  - The loop ends on the model's final output or raises ``MaxTurnsExceeded`` at the cap. No
    finish-on-files middleware is needed; finish is judged post-run by whether the outputs exist.
  - Per-turn ``model_call`` events come from ``RunHooks``; ``code_exec`` events from the shared
    ``run_python`` tool; token usage from ``result.context_wrapper.usage``.
  - Tracing reuses the Langfuse seam: the OpenInference instrumentor exports the SDK's spans to the
    active OTel provider, nested under the run's root span.
"""

from __future__ import annotations

from pathlib import Path

from agents import Agent, RunHooks, Runner, function_tool
from agents.exceptions import MaxTurnsExceeded

from .middleware import RunRecorder
from .models import model_usage_from_counts
from .netguard import NetPolicy
from .tools import RUN_PYTHON_DESCRIPTION, execute_python

_instrumented = False


class ProcessHooks(RunHooks):
    """Record one ``model_call`` event per LLM turn for the Agent Process plane.

    ``code_exec`` events are recorded by the ``run_python`` tool itself. Here each model turn is
    tagged with whether it produced a tool call (``has_code``) or the final answer (``is_finish``).
    """

    def __init__(self, recorder: RunRecorder) -> None:
        self.recorder = recorder
        self._last: dict | None = None

    async def on_llm_end(self, context, agent, response) -> None:  # noqa: ANN001
        self.recorder.record_model_call(has_code=False, is_finish=False)
        self._last = self.recorder.events[-1]

    async def on_tool_start(self, context, agent, tool) -> None:  # noqa: ANN001
        if self._last is not None:
            self._last["details"]["has_code"] = True

    async def on_agent_end(self, context, agent, output) -> None:  # noqa: ANN001
        if self._last is not None:
            self._last["details"]["is_finish"] = True


def _run_python_tool(
    workspace: Path,
    recorder: RunRecorder,
    timeout: int,
    net: "NetPolicy | None" = None,
    net_log=None,
    execution_backend: str = "local",
    docker_image: str | None = None,
):
    """Wrap the shared ``execute_python`` core as an OpenAI ``function_tool``."""

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
    return function_tool(run_python)


def build_openai_agent(
    model: str, *, workspace: Path, recorder: RunRecorder, system_prompt: str,
    run_python_timeout: int = 300, net: "NetPolicy | None" = None, net_log=None,
    execution_backend: str = "local", docker_image: str | None = None,
) -> Agent:
    """Build the OpenAI Agents SDK agent with the run_python tool and task instructions as system prompt.

    The SDK reads ``OPENAI_API_KEY`` from the environment. ``net`` is the harness-level network policy
    applied to each tool execution.
    """
    tool = _run_python_tool(
        Path(workspace).resolve(),
        recorder,
        run_python_timeout,
        net=net,
        net_log=net_log,
        execution_backend=execution_backend,
        docker_image=docker_image,
    )
    return Agent(name="biopulse", instructions=system_prompt, tools=[tool], model=model)


def _usage_to_model_usage(usage, model_key: str) -> dict:
    """Map ``RunResult.context_wrapper.usage`` onto biopulse-core's cost contract.

    OpenAI reports total ``input_tokens`` (cached reads are a subset, in
    ``input_tokens_details.cached_tokens``) and has no cache-write charge. This matches the
    TOTAL-input convention ``cost_summary`` expects, so cache_creation is 0.
    """
    if usage is None:
        return model_usage_from_counts(model_key, input_tokens=0, output_tokens=0, source_ref="openai-agents.usage")
    details = getattr(usage, "input_tokens_details", None)
    cached = 0
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
        if cached is None and isinstance(details, dict):
            cached = details.get("cached_tokens", 0)
    return model_usage_from_counts(
        model_key,
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read=int(cached or 0),
        cache_creation=0,
        source_ref="openai-agents.usage",
    )


def run_openai_agent(
    agent: Agent, user_message: str, *, recorder: RunRecorder, max_iterations: int, model_key: str
) -> tuple[dict, bool]:
    """Drive the Agents SDK loop; return (model_usage, hit_cap).

    ``max_iterations`` is the loop cap (SDK ``max_turns``). Exceeding it raises ``MaxTurnsExceeded``,
    caught here so the run is still scored on whatever outputs exist.
    """
    hooks = ProcessHooks(recorder)
    usage = None
    hit_cap = False
    try:
        result = Runner.run_sync(agent, user_message, max_turns=max_iterations, hooks=hooks)
        usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
    except MaxTurnsExceeded:
        hit_cap = True
    return _usage_to_model_usage(usage, model_key), hit_cap


def instrument_langfuse() -> None:
    """Route the SDK's OTel spans to the active (Langfuse) tracer provider, once per process."""
    global _instrumented
    if _instrumented:
        return
    try:
        from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

        OpenAIAgentsInstrumentor().instrument()
        _instrumented = True
    except Exception as exc:  # tracing must never fail the run
        print(f"[openai-agents] tracing instrumentation skipped: {exc}")
