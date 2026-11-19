"""Shared math prompt for trajectory training and response generation."""

MATH_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
INSTRUCTION_PLACEMENTS = ("system_prefix", "user_suffix")


def inference_instruction_placement(model_name: str, placement: str | None = None) -> str:
    if placement is None:
        if model_name.rstrip("/").rsplit("/", 1)[-1].lower() == "deepseek-math-7b-instruct":
            return "user_suffix"
        return "system_prefix"
    if placement not in INSTRUCTION_PLACEMENTS:
        raise ValueError(f"Unknown instruction placement: {placement}")
    return placement


def build_prompt(tokenizer, problem: str, system_prompt: str | None = None, *,
                 instruction_placement: str = "system_prefix") -> str:
    instruction = MATH_SYSTEM_PROMPT if system_prompt is None else system_prompt
    if instruction_placement == "user_suffix":
        messages = [{"role": "user", "content": f"{problem}\n{instruction}"}]
    else:
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": problem},
        ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
