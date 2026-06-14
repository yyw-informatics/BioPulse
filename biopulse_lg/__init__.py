"""BioPulse: a LangChain agent evaluation harness backed by LangGraph.

Provides the agent loop and tracing. The evaluation core (task registry,
scorers, benchmark packs, dual-plane scoring) is imported from
`biopulse`, a separately maintained editable dependency, rather than reimplemented:

    from biopulse.tasks.registry import get, required_outputs
    record = get("label_projection")
    result = record.scorer(benchmark_dir, run_dir)        # Scientific Artifact plane (dual-plane dict)
    from biopulse.runner.process_plane import summarize_process   # Agent Process plane
    from biopulse.runner.evidence import copy_public_to_workspace # workspace setup

The harness builds a tool-calling agent with LangChain 1.x ``create_agent``,
a ``run_python`` tool, finish-on-files middleware, and optional Langfuse or
LangSmith tracing.
"""

__version__ = "0.1.0"
