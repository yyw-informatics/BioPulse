"""End-to-end smoke test for the LangChain agent harness, offline.

Drives the real graph (``create_agent`` + middleware + the ``run_python`` subprocess) and the real
``biopulse-core`` scorer against the real label-projection pack, swapping the LLM for a scripted
fake model that emits one ``run_python`` tool call writing a majority-class prediction, then stops.
Exercises finish-on-files (``FinishOnOutputs`` jumps to END once the outputs exist), the
process-event recorder, the subprocess tool, and both scoring planes.

Set ``BIOPULSE_PACK`` to override the pack path; defaults to the standard build location and skips
if absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from biopulse.runner.evidence import copy_public_to_workspace
from biopulse.tasks.registry import required_outputs

from biopulse_lg.agent import build_agent
from biopulse_lg.middleware import RunRecorder
from biopulse_lg.score import score_process, score_scientific

_DEFAULT_PACK = (
    Path(os.environ.get("BIOPULSE_BENCHMARK_ROOT", "../biopulse-core/benchmark_packs"))
    / "op_label_projection_mini"
)

# Majority-class baseline standing in for the agent's solution. Writes a schema-valid
# prediction.h5ad (obs['label_pred'] for every test cell, in solution row order) and report.md.
_SOLUTION_CODE = """
import os
import anndata as ad

train = ad.read_h5ad("input/train.h5ad")
test = ad.read_h5ad("input/test.h5ad")
col = "label" if "label" in train.obs else "cell_type"
majority = str(train.obs[col].value_counts().idxmax())
test.obs["label_pred"] = [majority] * test.n_obs
test.uns["method_id"] = "smoke_majority"
os.makedirs("outputs", exist_ok=True)
test.write_h5ad("outputs/prediction.h5ad")
with open("outputs/report.md", "w") as fh:
    fh.write("# Smoke test\\nMajority-class baseline used by the smoke test.\\n")
print("wrote prediction for", test.n_obs, "cells; majority label:", majority)
"""


class _ScriptedToolModel(GenericFakeChatModel):
    """Fake chat model returning scripted messages, plus a minimal ``bind_tools`` implementation.

    ``create_agent`` calls ``bind_tools``, which the base ``GenericFakeChatModel`` does not implement.
    """

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ARG002
        return self


@pytest.fixture
def pack() -> Path:
    path = Path(os.environ.get("BIOPULSE_PACK", _DEFAULT_PACK))
    if not (path / "public").exists():
        pytest.skip(f"benchmark pack not found at {path}; set BIOPULSE_PACK to run this test")
    return path


def test_vertical_slice(pack: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    workspace = run_dir / "workspace"
    copy_public_to_workspace(pack, workspace)
    (workspace / "outputs").mkdir(parents=True, exist_ok=True)

    required = required_outputs("label_projection")
    recorder = RunRecorder()
    model = _ScriptedToolModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "run_python", "args": {"code": _SOLUTION_CODE}, "id": "call_1", "type": "tool_call"}],
                ),
                AIMessage(content="The prediction and report are written."),
            ]
        )
    )
    agent = build_agent(
        model,
        workspace=workspace,
        required_outputs=required,
        recorder=recorder,
        system_prompt="You are a test agent.",
        provider="openai",  # skips the Anthropic caching middleware in this offline test
        max_iterations=6,
    )

    agent.invoke({"messages": [{"role": "user", "content": "Solve the task."}]})

    # finish-on-files fired and the deliverables exist
    assert recorder.finished is True
    assert (workspace / "outputs/prediction.h5ad").exists()
    assert (workspace / "outputs/report.md").exists()

    # ProcessEvents counts the finishing turn (no off-by-one): turn 1 emits the run_python
    # tool call, turn 2 is the final message after the outputs exist.
    assert recorder.iterations == 2

    # run_python subprocess ran once and succeeded
    code_execs = [e for e in recorder.events if e["event_type"] == "code_exec"]
    assert len(code_execs) == 1
    assert code_execs[0]["details"]["ok"] is True
    assert code_execs[0]["details"]["timed_out"] is False

    # Scientific Artifact plane: real, schema-valid accuracy
    scientific = score_scientific("label_projection", pack, run_dir, run_dir.name)
    assert scientific["metrics"]["schema_valid"] == 1.0
    assert not scientific["violations"]
    accuracy = scientific["metrics"]["accuracy"]
    assert isinstance(accuracy, float) and 0.0 <= accuracy <= 1.0
    assert scientific["final_score"] == accuracy

    # Agent Process plane: behavioral summary matches the run
    process = score_process(recorder, finished=recorder.finished, max_iterations=6)
    assert process["finished_cleanly"] is True
    assert process["n_model_calls"] == 2  # both turns recorded, matching recorder.iterations
    assert process["n_code_execs"] == 1
    assert process["n_code_failures"] == 0
    assert process["hit_iteration_cap"] is False


def test_step_cap_stops_a_runaway(tmp_path: Path) -> None:
    """Verify ProcessEvents caps the loop when the model never writes the outputs."""
    workspace = tmp_path / "workspace"
    (workspace / "outputs").mkdir(parents=True, exist_ok=True)
    recorder = RunRecorder()
    # Endless no-op tool calls that never produce the required files.
    never_finishing = (
        AIMessage(
            content="",
            tool_calls=[{"name": "run_python", "args": {"code": "print('noop')"}, "id": f"c{i}", "type": "tool_call"}],
        )
        for i in range(1000)
    )
    model = _ScriptedToolModel(messages=never_finishing)
    agent = build_agent(
        model,
        workspace=workspace,
        required_outputs=required_outputs("label_projection"),
        recorder=recorder,
        system_prompt="test",
        provider="openai",
        max_iterations=4,
    )

    agent.invoke({"messages": [{"role": "user", "content": "go"}]})

    assert recorder.finished is False
    assert recorder.iterations == 4  # capped at max_iterations
    process = score_process(recorder, finished=False, max_iterations=4)
    assert process["hit_iteration_cap"] is True


def test_run_task_end_to_end(pack, tmp_path, monkeypatch):
    """Drive run_task end-to-end with a monkeypatched scripted model, so no Anthropic call.

    Exercises workspace staging, the real scorer, and artifact writing, which the unit tests above skip.
    """
    from biopulse_lg import run as run_module

    def _scripted_model(*args, **kwargs):
        return _ScriptedToolModel(
            messages=iter(
                [
                    AIMessage(content="", tool_calls=[{"name": "run_python", "args": {"code": _SOLUTION_CODE}, "id": "c1", "type": "tool_call"}]),
                    AIMessage(content="The prediction and report are written."),
                ]
            )
        )

    monkeypatch.setattr(run_module, "make_chat_model", _scripted_model)
    monkeypatch.setenv("BIOPULSE_TRACER", "none")  # disable tracing to keep the test hermetic

    result = run_module.run_task(
        "label_projection", pack, "claude-sonnet-4-6",
        run_id="pytest-e2e", runs_dir=tmp_path, max_iterations=6, overwrite=True,
    )

    assert result["metrics"]["schema_valid"] == 1.0
    assert result["passed"] is True
    run_dir = tmp_path / "pytest-e2e"
    for artifact in ("run_manifest.json", "evaluator_results.json", "process_summary.json", "cost_summary.json", "evidence_bundle.json"):
        assert (run_dir / artifact).exists(), f"missing artifact: {artifact}"
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["finished"] is True
    assert manifest["iterations"] == 2  # run.py records both model turns, including the finishing one
    assert manifest["hit_iteration_cap"] is False
