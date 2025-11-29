import tiktoken

def count_gpt_tokens(text: str, model: str = "gpt-5-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def estimate_gpt_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-5-mini",
    input_price_per_million: float = 0.25,  # $ per 1M input tokens
    output_price_per_million: float = 2.00  # $ per 1M output tokens
) -> float:
    input_cost = (input_tokens / 1_000_000) * input_price_per_million
    output_cost = (output_tokens / 1_000_000) * output_price_per_million
    return input_cost + output_cost
