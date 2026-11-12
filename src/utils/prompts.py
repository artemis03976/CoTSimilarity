"""Shared math prompt for trajectory training and response generation."""

MATH_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompt(tokenizer, problem: str, system_prompt: str = MATH_SYSTEM_PROMPT) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
