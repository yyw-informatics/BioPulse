"""Assemble the tool-calling agent with LangChain 1.x ``create_agent``.

``create_agent`` compiles a LangGraph runtime with a single ``run_python`` tool plus the
harness middleware. Finish-on-files and the step cap are middleware hooks rather than a
custom reinvocation loop. The caching middleware is registered first so it tags the system
and tool blocks before later hooks run.
"""

from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent

from .middleware import FinishOnOutputs, ProcessEvents, RunRecorder
from .models import caching_middleware
from .tools import make_run_python_tool


def build_agent(
    model,
    *,
    workspace: Path,
    required_outputs: list[str],
    recorder: RunRecorder,
    system_prompt: str,
    provider: str = "anthropic",
    max_iterations: int = 20,
    run_python_timeout: int = 300,
    net=None,
    net_log=None,
    execution_backend: str = "local",
    docker_image: str | None = None,
):
    """Build the compiled agent graph from model, ``run_python``, and middleware.

    ``model`` is a constructed chat model with the cost callback already attached. Returns the
    compiled graph; ``.invoke({"messages": [...]})`` drives it to completion or the step cap.
    """
    tool = make_run_python_tool(
        workspace,
        recorder,
        timeout=run_python_timeout,
        net=net,
        net_log=net_log,
        execution_backend=execution_backend,
        docker_image=docker_image,
    )
    # create_agent runs after_model hooks in REVERSE registration order (inner->outer), and the
    # first hook returning {"jump_to": "end"} short-circuits the rest. ProcessEvents must run before
    # FinishOnOutputs so the finishing turn is still counted, so it is registered LAST to execute first.
    middleware = [
        *caching_middleware(provider),  # outermost: tag system + tools for prompt caching
        FinishOnOutputs(workspace, required_outputs, recorder),
        ProcessEvents(recorder, max_iterations=max_iterations),
    ]
    return create_agent(model, tools=[tool], system_prompt=system_prompt, middleware=middleware)
