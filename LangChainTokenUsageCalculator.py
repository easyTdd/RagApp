from typing import Any, Iterable, Dict, Optional


class LangChainTokenUsageCalculator:
    """Calculator specialized for LangChain agent message objects.

    Public API:
    - LangChainTokenUsageCalculator(model)
    - compute(result_messages) -> token_usage dict

    Skaičiuoja:
    - bendrus input tokenus
    - iš jų cache_read (cached) ir ne-cache'intus
    - output tokenus
    - kainą atskirai už cached, ne-cached ir output

    Pastaba:
    - remiasi tik usage_metadata / token_usage, kuriuos grąžina LLM/LangChain.
    """

    _MODEL_COSTS = {
        # kainos $ per 1,000,000 tokenų
        "gpt-5-mini": {
            "input": 0.25,
            "input_cached": 0.25,  # jei OpenAI turi kitą kainą cache'ui – pakeisk čia
            "output": 2.00,
        },
        "gpt-4o": {
            "input": 0.50,
            "input_cached": 0.50,
            "output": 4.00,
        },
    }

    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model

    def _aggregate_usage_from_messages(
        self, result_messages: Iterable[Any]
    ) -> Dict[str, int]:
        """Surenka usage_metadata iš visų žinučių ir grąžina agreguotus skaičius.

        Tikimasi LangChain stiliaus:
        msg.usage_metadata = {
            "input_tokens": ...,
            "output_tokens": ...,
            "total_tokens": ...,
            "input_token_details": {"cache_read": ...},
            ...
        }

        Taip pat palaikomas dict variantas su 'usage_metadata' arba 'token_usage' key.
        """
        total_input = 0
        total_output = 0
        total_tokens = 0
        total_cached = 0

        for msg in result_messages:
            usage: Optional[Dict[str, Any]] = None

            # 1) LangChain message object: msg.usage_metadata
            if hasattr(msg, "usage_metadata"):
                usage = getattr(msg, "usage_metadata") or None

            # 2) Plain dict variant
            if usage is None and isinstance(msg, dict):
                usage = msg.get("usage_metadata") or msg.get("token_usage")

            if not usage:
                continue

            # Basic counts (support both usage_metadata and raw token_usage shapes)
            input_tokens = (
                usage.get("input_tokens")
                or usage.get("prompt_tokens")
                or 0
            )
            output_tokens = (
                usage.get("output_tokens")
                or usage.get("completion_tokens")
                or 0
            )
            this_total = usage.get("total_tokens") or (input_tokens + output_tokens)

            total_input += input_tokens
            total_output += output_tokens
            total_tokens += this_total

            # Cache details, jei yra
            input_details = usage.get("input_token_details") or {}
            cached = input_details.get("cache_read", 0)
            total_cached += cached

        return {
            "input_tokens_total": total_input,
            "input_tokens_cached": total_cached,
            "input_tokens_noncached": max(total_input - total_cached, 0),
            "output_tokens": total_output,
            "total_tokens": total_tokens,
        }

    def _estimate_gpt_cost(
        self,
        input_tokens_total: int,
        input_tokens_cached: int,
        output_tokens: int,
    ) -> dict:
        costs = self._MODEL_COSTS.get(self.model)
        if costs is None:
            costs = self._MODEL_COSTS["gpt-5-mini"]

        input_price_per_million = costs.get("input", 0.0)
        input_cached_price_per_million = costs.get(
            "input_cached", input_price_per_million
        )
        output_price_per_million = costs.get("output", 0.0)

        noncached_tokens = max(input_tokens_total - input_tokens_cached, 0)

        input_cost_cached = (
            input_tokens_cached / 1_000_000
        ) * input_cached_price_per_million
        input_cost_noncached = (
            noncached_tokens / 1_000_000
        ) * input_price_per_million
        input_cost_total = input_cost_cached + input_cost_noncached

        output_cost = (output_tokens / 1_000_000) * output_price_per_million
        total_cost = input_cost_total + output_cost

        return {
            "input_cost_total": input_cost_total,
            "input_cost_cached": input_cost_cached,
            "input_cost_noncached": input_cost_noncached,
            "output_cost": output_cost,
            "total_cost": total_cost,
        }

    def compute(self, result_messages: Iterable[Any]):
        """Pagrindinė funkcija.

        Grąžina dict su:
        - token counters (input_total, cached, noncached, output, total)
        - cost breakdown (input_cached, input_noncached, output, total)
        """
        aggregated = self._aggregate_usage_from_messages(result_messages)

        input_total = aggregated["input_tokens_total"]
        input_cached = aggregated["input_tokens_cached"]
        input_noncached = aggregated["input_tokens_noncached"]
        output_tokens = aggregated["output_tokens"]
        total_tokens = aggregated["total_tokens"]

        costs = self._estimate_gpt_cost(
            input_tokens_total=input_total,
            input_tokens_cached=input_cached,
            output_tokens=output_tokens,
        )

        return {
            # tokenai
            "input_tokens": input_total,
            "input_tokens_cached": input_cached,
            "input_tokens_noncached": input_noncached,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            # kainos
            "input_cost": costs["input_cost_total"],
            "input_cost_cached": costs["input_cost_cached"],
            "input_cost_noncached": costs["input_cost_noncached"],
            "output_cost": costs["output_cost"],
            "total_cost": costs["total_cost"],
        }
