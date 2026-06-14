"""CLI for running one task through the agent harness (LangChain ``create_agent`` or the
OpenAI Agents SDK), score it on both planes, and attach the verdict to the run's trace.

    python -m biopulse_lg.run \
        --task label_projection \
        --benchmark ../biopulse-core/benchmark_packs/op_label_projection_mini \
        --model claude-sonnet-4-6

The run-dir layout mirrors ``biopulse.runner.run_baseline`` (``runs/<run_id>/workspace/`` with
inputs copied from the pack's ``public/`` and outputs under ``workspace/outputs/``), so the core's
scorer and token/cost contract read it unchanged. The agent loop can run through
LangChain ``create_agent`` (default) or the OpenAI Agents SDK
(``--engine openai-agents``) — and adds pluggable tracing (Langfuse or LangSmith; see ``tracing.py``)
with the dual-plane scores attached. The model provider's API key is required (``ANTHROPIC_API_KEY``
by default, ``OPENAI_API_KEY`` for the openai-agents engine); tracing is optional and no-ops when no
backend is configured.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from biopulse.runner.cost import write_runtime_artifacts
from biopulse.runner.harness import harness_policy
from biopulse.runner.evidence import (
    copy_public_to_workspace,
    evidence_bundle,
    list_files,
    utc_now,
    write_json,
)
from biopulse.tasks.registry import TaskRecord
from biopulse.tasks.registry import get as get_task
from biopulse.tasks.registry import required_outputs

from .agent import build_agent
from .behavior import summarize_behavior
from .execution import DEFAULT_DOCKER_IMAGE
from .middleware import RunRecorder
from .netguard import NetPolicy
from .placement import load_boards, placement
from .models import LedgerCallbackHandler, infer_provider, make_chat_model, model_usage_from_counts
from .score import score_process, score_scientific
from .tracing import select_tracer

def _system_prompt(max_iterations: int) -> str:
    return f"""You are BioPulse, an autonomous agent solving a data-analysis task.

You have ONE tool: run_python(code) — it executes Python in your workspace directory and returns the
exit code, stdout, and stderr. Use it for everything.

You have a STRICT budget of about {max_iterations} tool calls, after which the run is terminated. Your
single most important goal is to PRODUCE THE REQUIRED OUTPUT FILES. A run that writes a solid, simple
solution scores; a run that explores forever and never writes the files scores ZERO.

Recommended approach:
1. (one call) Briefly inspect the data — shapes, obs columns, available features (e.g. obsm['X_pca'],
   layers['normalized']), and the training label distribution.
2. (one or two calls) Train ONE reasonable model. Do NOT run cross-validation sweeps or benchmark many
   classifiers — a single well-chosen model (e.g. logistic regression or kNN on the PCA embedding) is
   expected and sufficient.
3. Immediately WRITE the required outputs under outputs/ (the data file AND outputs/report.md). The run
   ends automatically the moment those files exist — do not narrate completion, just write them.

Rules:
- Inputs are under `input/`; write deliverables under `outputs/`.
- numpy, anndata, scikit-learn, and scipy are importable. Each run_python call is a FRESH process with
  no shared state — persist anything you need to files.
- Do not read anything under `hidden/`, do not look for solution files, and do not access the network.
"""


def _build_messages(record: TaskRecord, workspace: Path, max_iterations: int) -> tuple[str, str]:
    """Return (system_prompt, first_user_message). The user turn carries the task statement, the
    output schema, and a listing of the workspace contents."""
    listing = "\n".join(f"- {path}" for path in list_files(workspace)) or "(empty)"
    user = f"{record.instruction}\n\n{record.output_schema}\n\n## Files in your workspace\n{listing}\n"
    return _system_prompt(max_iterations), user


def _tracing_line(tracer, trace_id) -> str:
    if not tracer.enabled:
        return "off (no tracer configured)"
    url = tracer.url(trace_id)
    return f"{tracer.name} → {url}" if url else f"{tracer.name} (trace {trace_id})"


def _format_method_menu(menu: dict) -> str:
    lines = []
    for method in (menu or {}).get("methods", []):
        summary = (method.get("summary") or "").strip()
        lines.append(f"- {method.get('id')} ({method.get('label')}): {summary}")
    return "\n".join(lines)


_L4_RESEARCH = (
    "\n\n## Required: method research and fitness analysis (do this first)\n"
    "Before writing the solution, evaluate the candidate methods in outputs/report.md: for the leading "
    "options note their assumptions, strengths, and limitations, and assess each method's fitness for THIS "
    "dataset specifically (its size, dimensionality, label distribution, batch structure). Then choose ONE "
    "method, justify it against the alternatives, and implement it. Record the chosen method in uns[\"method_id\"]."
)


def _augment_for_level(user_message: str, methods_text: str | None, force_research: bool) -> str:
    """Append the L3 method menu and/or L4 research instruction to the task hand-off."""
    if methods_text:
        user_message += (
            "\n\n## Curated method menu (reference)\n"
            "Established methods benchmarked for this task are listed below and in methods.yaml in your "
            "workspace. You may use one of these or a method of your own choosing.\n" + methods_text + "\n"
        )
    if force_research:
        user_message += _L4_RESEARCH
    return user_message


def _resolve_level(level: str, benchmark: Path):
    """Map a harness level to (NetPolicy, method-menu text | None, force_research) from the core policy
    plus the pack's blacklist.yaml / methods.yaml. Missing pack files degrade with a warning."""
    policy = harness_policy(level)  # raises ValueError on an unknown level
    if policy["mode"] == "block_all":
        net = NetPolicy("block_all", ())
    else:
        hosts = set(policy["blacklist"])  # canonical answer-source blacklist from biopulse-core
        blacklist_file = benchmark / "blacklist.yaml"
        if blacklist_file.exists():
            hosts |= set((yaml.safe_load(blacklist_file.read_text(encoding="utf-8")) or {}).get("hosts", []) or [])
        else:
            print(f"[level] {level}: pack has no blacklist.yaml — using the core contamination blacklist only")
        net = NetPolicy("blacklist", tuple(sorted(hosts)))
    methods_text = None
    if policy["injected_resources"]:
        methods_file = benchmark / "methods.yaml"
        if methods_file.exists():
            methods_text = _format_method_menu(yaml.safe_load(methods_file.read_text(encoding="utf-8")))
        else:
            print(f"[level] {level}: pack has no methods.yaml — method menu unavailable for L3/L4")
    return net, methods_text, policy["force_research"]


def _place_on_leaderboard(benchmark: Path, scientific: dict, workspace: Path, output_filename: str):
    """Place the run on the pack's OP leaderboard (if present) using the board's primary metric.
    Reads the agent's self-reported method from the output file's ``uns['method_id']``. No reruns."""
    leaderboard, menu_ids = load_boards(benchmark)
    if leaderboard is None:
        return None
    metric = leaderboard.get("dataset", {}).get("metric_primary", "accuracy")
    score = (scientific.get("metrics") or {}).get(metric)
    if score is None:
        return None
    method_id = None
    output = Path(workspace) / "outputs" / output_filename
    if output.exists():
        try:
            import anndata as ad

            value = ad.read_h5ad(output, backed="r").uns.get("method_id")
            method_id = str(value) if value is not None else None
        except Exception:
            method_id = None
    return placement(float(score), method_id, leaderboard, menu_ids=menu_ids)


def _fold_network_events(net_log, recorder) -> None:
    """Fold the netguard audit log into recorder.events as web_fetch events so summarize_process
    reports n_web_fetches / n_blocked_fetches. Call once, after the agent loop."""
    path = Path(net_log) if net_log else None
    if not path or not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        recorder.events.append(
            {"event_type": "web_fetch", "details": {"blocked": bool(entry.get("blocked")), "host": entry.get("host")}}
        )


def run_task(
    task: str,
    benchmark: Path | str,
    model: str | None = None,
    *,
    engine: str = "langchain",
    level: str = "L1",
    run_id: str | None = None,
    runs_dir: Path | str = "runs",
    max_iterations: int = 20,
    run_python_timeout: int = 300,
    execution_backend: str = "local",
    docker_image: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Drive one agent run (LangChain ``create_agent`` or the OpenAI Agents SDK) and return the
    scientific-plane scorer result."""
    load_dotenv()
    benchmark = Path(benchmark).expanduser()
    record = get_task(task)
    task_type = record.task_type
    required = required_outputs(task_type)
    # Default model and cost-table provider per engine.
    model = model or ("gpt-4o-mini" if engine == "openai-agents" else "claude-sonnet-4-6")
    provider = "openai" if engine == "openai-agents" else infer_provider(model)
    agent_id = f"{'openai-agents' if engine == 'openai-agents' else 'create_agent'}/{model}"
    model_key = f"{provider}/{model}"

    run_id = run_id or f"{task_type}-{model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # Resolve to absolute: the run_python subprocess (cwd + script path) and the scorer all require an
    # unambiguous location independent of the CLI's launch directory.
    run_dir = (Path(runs_dir) / run_id).resolve()
    workspace = run_dir / "workspace"
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Run already exists: {run_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Stage the workspace as the core expects, then pre-create outputs/ so the agent's
    # ad.write_h5ad("outputs/...") does not fail on a missing directory.
    copy_public_to_workspace(benchmark, workspace)
    (workspace / "outputs").mkdir(parents=True, exist_ok=True)
    # Harness level: network policy, plus the curated method menu (L3/L4) and forced research step (L4).
    net, methods_text, force_research = _resolve_level(level, benchmark)
    net_log = run_dir / "network_audit.jsonl"
    docker_image_effective = (
        docker_image or os.environ.get("BIOPULSE_DOCKER_IMAGE") or DEFAULT_DOCKER_IMAGE
        if execution_backend == "docker-none"
        else None
    )
    if methods_text is not None:  # L3/L4: stage the method menu from pack root into the agent's workspace
        shutil.copy2(benchmark / "methods.yaml", workspace / "methods.yaml")
    files_available = list_files(workspace)

    recorder = RunRecorder()
    system_prompt, user_message = _build_messages(record, workspace, max_iterations)
    user_message = _augment_for_level(user_message, methods_text, force_research)
    inputs = {"task": task_type, "benchmark": str(benchmark), "model": model, "engine": engine,
              "level": level, "run_id": run_id, "execution_backend": execution_backend,
              "docker_image": docker_image_effective}
    tracer = select_tracer()
    error: str | None = None
    trace_id = None
    model_usage: dict | None = None
    model_calls = 0
    start_utc = utc_now()
    started = time.perf_counter()

    with tracer.start_run("biopulse_run", inputs) as run:
        trace_id = run.trace_id
        try:
            if engine == "openai-agents":
                from .openai_agent import build_openai_agent, instrument_langfuse, run_openai_agent

                if tracer.name == "langfuse":
                    instrument_langfuse()  # export the SDK's spans into the same Langfuse trace
                agent = build_openai_agent(
                    model, workspace=workspace, recorder=recorder,
                    system_prompt=system_prompt, run_python_timeout=run_python_timeout,
                    net=net, net_log=net_log, execution_backend=execution_backend,
                    docker_image=docker_image_effective,
                )
                model_usage, _ = run_openai_agent(
                    agent, user_message, recorder=recorder,
                    max_iterations=max_iterations, model_key=model_key,
                )
                model_calls = recorder.iterations
            else:
                ledger = LedgerCallbackHandler(model=model, provider=provider)
                chat_model = make_chat_model(model, provider=provider, callbacks=[ledger])
                agent = build_agent(
                    chat_model, workspace=workspace, required_outputs=required, recorder=recorder,
                    system_prompt=system_prompt, provider=provider, max_iterations=max_iterations,
                    run_python_timeout=run_python_timeout, net=net, net_log=net_log,
                    execution_backend=execution_backend, docker_image=docker_image_effective,
                )
                config = {"callbacks": run.callbacks} if run.callbacks else None
                agent.invoke({"messages": [{"role": "user", "content": user_message}]}, config=config)
                model_usage, model_calls = ledger.model_usage(), ledger.calls
        except Exception as exc:  # keep partial artifacts and score what exists
            error = f"{type(exc).__name__}: {exc}"
            print(f"[agent] run failed: {error}")
        if model_usage is None:
            model_usage = model_usage_from_counts(model_key, input_tokens=0, output_tokens=0, source_ref="none")

    wall_time = time.perf_counter() - started
    end_utc = utc_now()

    _fold_network_events(net_log, recorder)  # netguard audit log -> web_fetch events for the process plane

    outputs_exist = all((workspace / rel).exists() for rel in required)
    finished = recorder.finished or outputs_exist

    scientific = score_scientific(task_type, benchmark, run_dir, run_id)
    process = score_process(recorder, finished=finished, max_iterations=max_iterations)
    placement_result = _place_on_leaderboard(benchmark, scientific, workspace, record.output.filename)
    behavior = None
    try:
        behavior = summarize_behavior(recorder.events)
    except Exception as exc:  # behavior is a bonus artifact; never fail the run over it
        print(f"[behavior] summary skipped: {exc}")

    files_after = list_files(workspace)
    manifest = {
        "schema_version": "biopulse.run_manifest.v1",
        "run_id": run_id,
        "task_id": record.task_id,
        "task_type": task_type,
        "agent_id": agent_id,
        "agent_surface": "openai_agents_sdk" if engine == "openai-agents" else "langgraph_create_agent",
        "engine": engine,
        "level": level,
        "model": model,
        "provider": provider,
        "execution_backend": execution_backend,
        "docker_image": docker_image_effective,
        "benchmark": str(benchmark),
        "workspace": str(workspace),
        "start_time_utc": start_utc,
        "end_time_utc": end_utc,
        "wall_time_seconds": wall_time,
        "iterations": recorder.iterations,
        "max_iterations": max_iterations,
        "finished": finished,
        "hit_iteration_cap": process["hit_iteration_cap"],
        "model_calls": model_calls,
        "error": error,
        "files_available_to_agent": files_available,
        "files_produced_by_agent": sorted(p for p in files_after if p not in files_available),
        "hidden_ground_truth_excluded": (
            "hidden" not in {part.lower() for path in files_after for part in Path(path).parts}
            and "solution.h5ad" not in {Path(path).name.lower() for path in files_after}
        ),
    }

    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "evaluator_results.json", scientific)
    write_json(run_dir / "process_summary.json", process)
    write_json(run_dir / "evidence_bundle.json", evidence_bundle(manifest, scientific))
    if placement_result is not None:
        write_json(run_dir / "placement.json", placement_result)
    if behavior is not None:
        write_json(run_dir / "behavior_summary.json", behavior)
    cost = write_runtime_artifacts(
        run_dir, run_id, agent_id=agent_id, agent_kind="llm_agent",
        model=model_key, provider=provider, model_usage=model_usage,
    )

    if tracer.enabled:
        tracer.attach_scores(trace_id, scientific=scientific, process=process)

    metrics = scientific.get("metrics", {})
    print(
        f"\n  {run_id}  [{level}]\n"
        f"    scientific : passed={scientific['passed']} final_score={scientific['final_score']} "
        f"accuracy={metrics.get('accuracy')} schema_valid={metrics.get('schema_valid')} "
        f"violations={len(scientific.get('violations', []))}\n"
        f"    process    : iterations={process['iterations']}/{process['max_iterations']} "
        f"finished_cleanly={process['finished_cleanly']} code_execs={process['n_code_execs']} "
        f"code_error_rate={process['code_error_rate']} "
        f"web={process['n_web_fetches']} blocked={process['n_blocked_fetches']}\n"
        f"    cost       : ${cost['total_cost_usd']} "
        f"(in={cost['total_input_tokens']} out={cost['total_output_tokens']} tokens, {model_calls} calls)\n"
        f"    artifacts  : {run_dir}\n"
        f"    tracing    : {_tracing_line(tracer, trace_id)}\n"
    )
    if behavior is not None:
        print(
            f"    behavior   : errors={behavior['error_types']} recovered={behavior['recovered']} "
            f"escapes={behavior['n_subprocess_escapes']} explore_execs={behavior['n_exploration_execs']} "
            f"first_output@{behavior['turns_to_first_output']} exec_s={behavior['total_exec_seconds']}"
        )
    if placement_result is not None:
        pr = placement_result
        metric = pr["metric"]
        worse, better = pr["nearest_worse"], pr["nearest_better"]
        neighbors = " / ".join(
            part for part in (
                f"worse: {worse['method_id']} ({worse[metric]:.3f})" if worse else None,
                f"better: {better['method_id']} ({better[metric]:.3f})" if better else None,
            ) if part
        )
        print(
            f"    placement  : OP {pr['dataset_id']} — rank {pr['rank']}/{pr['n_methods']} real methods "
            f"({metric}={pr['agent_score']:.4f}); {neighbors}\n"
        )
    return scientific


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a BioPulse task through the agent harness (LangChain or OpenAI Agents SDK)")
    parser.add_argument("--task", required=True, help="task type or alias, e.g. label_projection / lp")
    parser.add_argument("--benchmark", required=True, type=Path, help="path to a benchmark pack")
    parser.add_argument("--engine", choices=["langchain", "openai-agents"], default="langchain",
                        help="agent loop: LangChain create_agent (default) or the OpenAI Agents SDK")
    parser.add_argument("--level", choices=["L1", "L2", "L3", "L4"], default="L1",
                        help="harness level: L1 no network; L2 +web minus blacklist; L3 +method menu; L4 +forced research")
    parser.add_argument("--model", default=None, help="agent model id; default per engine (claude-sonnet-4-6 / gpt-4o-mini)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-dir", default="runs", type=Path)
    parser.add_argument("--max-iterations", type=int, default=20, help="our own loop cap (NOT recursion_limit)")
    parser.add_argument("--run-python-timeout", type=int, default=300, help="per run_python subprocess timeout (s)")
    parser.add_argument("--execution-backend", choices=["local", "docker-none"], default="local",
                        help="where run_python executes: local subprocess or Docker with --network none")
    parser.add_argument("--docker-image", default=None,
                        help=f"Docker image for --execution-backend docker-none (default: {DEFAULT_DOCKER_IMAGE})")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    result = run_task(
        args.task, args.benchmark, args.model, engine=args.engine, level=args.level,
        run_id=args.run_id, runs_dir=args.runs_dir,
        max_iterations=args.max_iterations, run_python_timeout=args.run_python_timeout,
        execution_backend=args.execution_backend, docker_image=args.docker_image,
        overwrite=args.overwrite,
    )
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
