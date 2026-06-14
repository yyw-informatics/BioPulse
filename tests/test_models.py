"""Unit tests for the cost ledger. It stores raw TOTAL input; biopulse-core's cost model subtracts
the cache itself. No network needed."""

from __future__ import annotations

from types import SimpleNamespace

from biopulse_lg.models import LedgerCallbackHandler, infer_provider


def _llm_result(usage):
    """Build a minimal LLMResult: ``.generations[[ChatGeneration]]`` with a message carrying
    ``usage_metadata``, matching what ``on_llm_end`` walks."""
    message = SimpleNamespace(usage_metadata=usage)
    generation = SimpleNamespace(message=message)
    return SimpleNamespace(generations=[[generation]])


def test_ledger_stores_total_input_and_recovers_ttl_cache_creation():
    handler = LedgerCallbackHandler(model="claude-sonnet-4-6", provider="anthropic")
    # Under AnthropicPromptCachingMiddleware the generic cache_creation is zeroed and the real count
    # lands in ephemeral_5m_input_tokens; the handler recovers it via the ttl-specific fallback.
    handler.on_llm_end(
        _llm_result(
            {
                "input_tokens": 1000,
                "output_tokens": 50,
                "input_token_details": {"cache_read": 200, "cache_creation": 0, "ephemeral_5m_input_tokens": 300},
            }
        )
    )
    usage = handler.model_usage()["anthropic/claude-sonnet-4-6"]
    assert usage["input_tokens"] == 1000  # raw TOTAL, not pre-split; cost.py subtracts the cache
    assert usage["input_tokens_cache_read"] == 200
    assert usage["input_tokens_cache_write"] == 300  # recovered from the ttl-specific key
    assert usage["output_tokens"] == 50
    assert usage["total_tokens"] == 1050
    assert usage["measurement_method"] == "observed"
    assert handler.calls == 1


def test_ledger_accumulates_across_calls_without_double_counting_cache():
    handler = LedgerCallbackHandler(model="claude-sonnet-4-6", provider="anthropic")
    handler.on_llm_end(
        _llm_result(
            {"input_tokens": 1000, "output_tokens": 50,
             "input_token_details": {"cache_read": 200, "ephemeral_5m_input_tokens": 300}}
        )
    )
    handler.on_llm_end(
        _llm_result(
            {"input_tokens": 500, "output_tokens": 20,
             "input_token_details": {"cache_read": 100, "cache_creation": 40}}  # generic key populated here
        )
    )
    usage = handler.model_usage()["anthropic/claude-sonnet-4-6"]
    assert usage["input_tokens"] == 1500
    assert usage["output_tokens"] == 70
    assert usage["input_tokens_cache_read"] == 300
    assert usage["input_tokens_cache_write"] == 340  # 300 ephemeral + 40 generic; fallback, never summed twice
    assert handler.calls == 2


def test_ledger_ignores_messages_without_usage_metadata():
    handler = LedgerCallbackHandler(model="claude-sonnet-4-6", provider="anthropic")
    handler.on_llm_end(_llm_result(None))
    usage = handler.model_usage()["anthropic/claude-sonnet-4-6"]
    assert usage["input_tokens"] == 0 and usage["output_tokens"] == 0
    assert handler.calls == 0


def test_infer_provider():
    assert infer_provider("claude-sonnet-4-6") == "anthropic"
    assert infer_provider("gpt-4o") == "openai"
    assert infer_provider("o3-mini") == "openai"
