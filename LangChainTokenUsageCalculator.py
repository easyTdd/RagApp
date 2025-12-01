import tiktoken
from typing import Any, Iterable


class LangChainTokenUsageCalculator:
    """Calculator specialized for LangChain agent message objects.

    Public API:
    - LangChainTokenUsageCalculator(model)
    - compute(result_messages, assistant_response_text, system_content) -> token_usage dict

    Notes:
    - Model pricing is internal to this class and chosen by `model` passed to the
      constructor. To add or change prices, update `_MODEL_COSTS` in this file.
    """

    _MODEL_COSTS = {
        # prices are $ per 1,000,000 tokens
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        # examples; adjust as needed
        "gpt-4o": {"input": 0.50, "output": 4.00},
    }

    def __init__(self, model: str = "gpt-5-mini"):
        # Accept a model name and use the internal cost table. If the model
        # isn't present, fallback to the default pricing (gpt-5-mini).
        self.model = model

    def _count_gpt_tokens(self, text: str) -> int:
        enc = tiktoken.encoding_for_model(self.model)
        return len(enc.encode(text))

    def _build_message_list(self, result_messages: Iterable[Any], system_content: str = ""):
        message_list = []
        if system_content:
            message_list.append({"role": "system", "content": system_content})

        for msg in result_messages:
            # If caller passed plain dicts, accept them
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                message_list.append({"role": msg.get("role"), "content": msg.get("content", "")})
                continue

            # Determine role from LangChain message object
            role = "assistant"
            t = getattr(msg, "type", None)
            if t == "human":
                role = "user"
            elif t == "ai":
                role = "assistant"
            elif t == "tool":
                role = "tool"
            elif t == "system":
                role = "system"
            else:
                role = "assistant"

            # Extract content
            content = getattr(msg, "content", None)
            if content is None:
                pretty = getattr(msg, "pretty_print", None)
                if callable(pretty):
                    from io import StringIO
                    from contextlib import redirect_stdout
                    buf = StringIO()
                    with redirect_stdout(buf):
                        try:
                            pretty()
                        except Exception:
                            pass
                    content = buf.getvalue()

            if content is None:
                content = ""

            message_list.append({"role": role, "content": content})

        return message_list

    def _calculate_token_usage(self, messages):
        input_tokens = 0
        output_tokens = 0
        current_prompt = []
        for msg in messages:
            current_prompt.append(msg)
            if msg.get("role") == "assistant":
                # Input is all messages up to (but excluding) this assistant message
                prompt_tokens = sum(self._count_gpt_tokens(m.get("content", "")) for m in current_prompt[:-1])
                input_tokens += prompt_tokens
                # Output is assistant message itself
                output_tokens += self._count_gpt_tokens(msg.get("content", ""))
        total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def _estimate_gpt_cost(self, input_tokens: int, output_tokens: int) -> dict:
        costs = self._MODEL_COSTS.get(self.model)
        if costs is None:
            costs = self._MODEL_COSTS["gpt-5-mini"]
        input_price_per_million = costs.get("input", 0.0)
        output_price_per_million = costs.get("output", 0.0)

        input_cost = (input_tokens / 1_000_000) * input_price_per_million
        output_cost = (output_tokens / 1_000_000) * output_price_per_million
        total_cost = input_cost + output_cost
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
        }

    def compute(self, result_messages, system_content: str = ""):
        """Public method: returns token usage and cost dict for given conversation.

        NOTE: This uses the canonical `result_messages` list only. Do not pass
        a duplicate assistant response separately; assistant messages should
        already be present in `result_messages` and will be counted from there.
        """
        message_list = self._build_message_list(result_messages, system_content=system_content)

        tok = self._calculate_token_usage(message_list)
        input_tokens = tok.get("input_tokens", 0)
        output_tokens = tok.get("output_tokens", 0)
        total_tokens = tok.get("total_tokens", 0)

        costs = self._estimate_gpt_cost(input_tokens, output_tokens)

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_cost": costs.get("input_cost", 0.0),
            "output_cost": costs.get("output_cost", 0.0),
            "total_cost": costs.get("total_cost", 0.0),
        }
