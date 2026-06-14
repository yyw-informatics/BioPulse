"""Offline tests for the OpenAI Agents SDK runner's process-event and usage mapping.

End-to-end behavior is covered separately by a live gpt-4o-mini run.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from biopulse_lg.middleware import RunRecorder
from biopulse_lg.openai_agent import ProcessHooks, _usage_to_model_usage


def test_process_hooks_map_model_and_tool_turns():
    recorder = RunRecorder()
    hooks = ProcessHooks(recorder)

    async def scenario():
        # Turn 1: model requests a tool call.
        await hooks.on_llm_end(None, None, None)
        await hooks.on_tool_start(None, None, None)  # tags turn 1 as has_code
        recorder.record_code_exec(ok=True, timed_out=False)  # recorded by the tool itself
        # Turn 2: model produces the final answer.
        await hooks.on_llm_end(None, None, None)
        await hooks.on_agent_end(None, None, "done")  # tags turn 2 as is_finish

    asyncio.run(scenario())

    model_calls = [e for e in recorder.events if e["event_type"] == "model_call"]
    assert recorder.iterations == 2 and len(model_calls) == 2
    assert model_calls[0]["details"] == {"has_code": True, "is_finish": False}
    assert model_calls[1]["details"] == {"has_code": False, "is_finish": True}
    code_execs = [e for e in recorder.events if e["event_type"] == "code_exec"]
    assert len(code_execs) == 1 and code_execs[0]["details"]["ok"] is True


def test_usage_to_model_usage_reports_total_input_and_cache_read():
    usage = SimpleNamespace(
        input_tokens=1000, output_tokens=50, input_tokens_details=SimpleNamespace(cached_tokens=200)
    )
    mu = _usage_to_model_usage(usage, "openai/gpt-4o-mini")["openai/gpt-4o-mini"]
    assert mu["input_tokens"] == 1000  # raw TOTAL — cost_summary subtracts the cached portion itself
    assert mu["input_tokens_cache_read"] == 200
    assert mu["input_tokens_cache_write"] == 0  # OpenAI has no cache-write charge
    assert mu["output_tokens"] == 50
    assert mu["measurement_method"] == "observed"


def test_usage_to_model_usage_handles_missing_usage():
    mu = _usage_to_model_usage(None, "openai/gpt-4o-mini")["openai/gpt-4o-mini"]
    assert mu["input_tokens"] == 0 and mu["output_tokens"] == 0
