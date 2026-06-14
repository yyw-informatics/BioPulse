"""Chat-model factory and a cost-capture callback feeding BioPulse's token/cost contract.

Accounting reads ``usage_metadata`` off each ``AIMessage`` in the 2-D ``generations`` grid,
including ``input_token_details.cache_read`` / ``cache_creation``. Cost contract:
``biopulse.runner.cost.cost_summary`` expects ``input_tokens`` to be the TOTAL input (cached
reads/writes are a subset) and subtracts the cached portion itself. We store raw and do not
pre-subtract; pre-splitting would discount the cache twice.

ChatAnthropic already folds ``cache_read + cache_creation`` back into ``input_tokens``
(langchain-anthropic chat_models.py), so the raw ``usage_metadata.input_tokens`` we store is
the TOTAL the cost model wants.
"""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

# 8192 keeps a long ``run_python`` argument or report.md write from truncating mid-tool-call.
_MAX_TOKENS = 8192


def infer_provider(model: str) -> str:
    """Map a bare model id to its provider (price-table key prefix and caching policy).

    Anthropic is the default provider; OpenAI model ids are supported for alternate runs.
    """
    name = model.lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    return "anthropic"


def make_chat_model(
    model: str,
    *,
    provider: str | None = None,
    callbacks: list | None = None,
    max_tokens: int = _MAX_TOKENS,
):
    """Construct a LangChain chat model for ``create_agent``.

    LangChain reads the API key from the environment (``ANTHROPIC_API_KEY`` /
    ``OPENAI_API_KEY`` after ``load_dotenv``), so secrets never pass through this code.
    """
    provider = provider or infer_provider(model)
    if provider == "anthropic":
        return ChatAnthropic(model=model, max_tokens=max_tokens, callbacks=callbacks)
    return ChatOpenAI(model=model, max_tokens=max_tokens, callbacks=callbacks)


def caching_middleware(provider: str) -> list:
    """Return Anthropic prompt-caching middleware (OpenAI caches server-side, so none needed).

    ``ttl='5m'`` matches the core ledger's 1.25x cache-write rate. Returned as a list so the
    caller can splat it first (outermost) into ``create_agent``.
    """
    if provider == "anthropic":
        return [AnthropicPromptCachingMiddleware(ttl="5m", unsupported_model_behavior="ignore")]
    return []


def model_usage_from_counts(
    model_key: str, *, input_tokens: int, output_tokens: int, cache_read: int = 0,
    cache_creation: int = 0, reasoning_tokens: int | None = None, source_ref: str,
) -> dict[str, dict[str, Any]]:
    """Build the biopulse-core ``model_usage`` record from token totals.

    ``input_tokens`` is the TOTAL (``cost_summary`` subtracts the cached portion itself); cache
    read/write are reported separately so the core prices them at its discounted rates.
    """
    return {
        model_key: {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(input_tokens) + int(output_tokens),
            "input_tokens_cache_read": int(cache_read),
            "input_tokens_cache_write": int(cache_creation),
            "reasoning_tokens": reasoning_tokens,
            "measurement_method": "observed",
            "estimator": None,
            "source_ref": source_ref,
        }
    }


class LedgerCallbackHandler(BaseCallbackHandler):
    """Accumulate per-run token usage from ``usage_metadata`` into the shape ``biopulse.runner.cost`` consumes.

    One handler per run with one known model, so everything keys under the price-table join key
    ``"<provider>/<model>"`` rather than parsing model names from each response. Call
    :meth:`model_usage` after the run and pass it to ``cost.write_runtime_artifacts``.
    """

    def __init__(self, *, model: str, provider: str):
        self.model_key = f"{provider}/{model}"
        self.calls = 0
        self._input = 0  # TOTAL input (cache read + creation included); see module docstring
        self._output = 0
        self._cache_read = 0
        self._cache_creation = 0
        self._reasoning = 0

    def on_llm_end(self, response, **kwargs: Any) -> None:  # noqa: ANN001
        # LLMResult.generations is a 2-D list[list[ChatGeneration]]; the nested loop is required.
        for generations in getattr(response, "generations", []) or []:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) if message else None
                if not usage:
                    continue
                details = usage.get("input_token_details", {}) or {}
                output_details = usage.get("output_token_details", {}) or {}
                self._input += int(usage.get("input_tokens", 0) or 0)
                self._output += int(usage.get("output_tokens", 0) or 0)
                cache_read = int(details.get("cache_read", 0) or 0)
                if cache_read == 0:
                    cache_read = sum(
                        int(value or 0) for key, value in details.items() if str(key).endswith("cache_read")
                    )
                self._cache_read += cache_read
                self._reasoning += int(output_details.get("reasoning", 0) or output_details.get("reasoning_tokens", 0) or 0)
                # AnthropicPromptCachingMiddleware(ttl=...) makes langchain-anthropic zero the
                # generic `cache_creation` key and report the count under ttl-specific keys, so fall
                # back to those. A fallback, not a sum: summing would double-count when the generic
                # key is the populated one.
                cache_creation = int(details.get("cache_creation", 0) or 0)
                if cache_creation == 0:
                    cache_creation = int(details.get("ephemeral_5m_input_tokens", 0) or 0) + int(
                        details.get("ephemeral_1h_input_tokens", 0) or 0
                    )
                self._cache_creation += cache_creation
                self.calls += 1

    def model_usage(self) -> dict[str, dict[str, Any]]:
        """Return the ``model_usage`` document for ``cost.token_usage`` / ``cost.cost_summary``."""
        return model_usage_from_counts(
            self.model_key, input_tokens=self._input, output_tokens=self._output,
            cache_read=self._cache_read, cache_creation=self._cache_creation,
            reasoning_tokens=(self._reasoning or None),
            source_ref="langchain.usage_metadata",
        )
