"""Prompt template for CoT dependency analysis."""

from typing import List, Dict, Tuple


SYSTEM_PROMPT = """You are a rigorous mathematical logic analysis engine. Your task is to analyze a given mathematical reasoning chain and construct an "Information Dependency Directed Acyclic Graph (DAG)" for this reasoning chain.

[Input Format]
- [0] Original problem (Prompt)
- [1] to [N] are model-generated reasoning steps (already aggregated)

[Core Task]
For each step from 1 to N, strictly based on the literal text, determine which prerequisite information this step's derivation **directly uses**.
Dependency sources can only be one of the following three types:
1. Previous step numbers (e.g., [1], [2]...): This step directly references numerical values or algebraic expressions calculated earlier.
2. [0]: This step directly extracts known conditions or objectives from the original problem.
3. [External]: This step introduces new formulas, theorems, or fabricated values that were not derived earlier (e.g., directly stating "according to the AM-GM inequality", "the area formula for a circle", etc.).

[Macro-Action Classification]
For each step, you must also assign ONE macro-action tag from the following finite set:
- [Define]: Define variables or extract known conditions from the problem statement
- [Recall]: Recall or introduce external formulas/theorems (typically used with [External] dependency)
- [Derive]: Algebraic derivation, equation transformation, symbolic manipulation
- [Calculate]: Pure numerical calculation (arithmetic operations)
- [Verify]: Self-check, verification, or validation of previous results
- [Conclude]: Draw intermediate or final conclusions

Rules for tag assignment:
1. Each step must have exactly ONE tag
2. [Recall] is typically paired with [External] dependency but not always
3. [Define] is typically paired with [0] dependency (extracting from problem)
4. Choose the PRIMARY action if a step involves multiple operations

[Strict Rules]
1. Only find **direct dependencies**. If Step 3 depends on Step 2, and Step 2 depends on Step 1, then for Step 3, only output [2], not [1].
2. Do not omit any step! If the input has N steps, your output array must contain exactly N objects.

[Output Format]
Must output a valid JSON array in the following format:
[
  {
    "step_id": 1,
    "analysis": "Brief reasoning (e.g., introduces external uncertainty principle formula)",
    "depends_on": ["External"],
    "macro_action_tag": "Recall"
  },
  {
    "step_id": 2,
    "analysis": "Brief reasoning (e.g., extracts the objective to find minimum S from the original problem)",
    "depends_on": [0],
    "macro_action_tag": "Define"
  },
  {
    "step_id": 3,
    "analysis": "Brief reasoning (e.g., substitutes formula from step 1 into expression from step 2)",
    "depends_on": [1, 2],
    "macro_action_tag": "Derive"
  }
]"""


USER_PROMPT_TEMPLATE = """[Reasoning Chain to Analyze]
[0] Problem Description: {problem}

{steps}

Please analyze the dependency relationships for all {num_steps} steps and output a JSON array."""


def format_steps(steps_list: List[Dict]) -> str:
    """Format steps list into numbered text.

    Args:
        steps_list: List of dicts with 'index' and 'text' keys

    Returns:
        Formatted string like "[1] Step 1: ...\n[2] Step 2: ..."
    """
    return "\n".join([
        f"[{step['index']}] Step {step['index']}: {step['text']}"
        for step in steps_list
    ])


def build_prompt(problem: str, steps: List[Dict]) -> Tuple[str, str]:
    """Build complete prompt for LLM.

    Args:
        problem: Problem statement string
        steps: List of step dicts

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    formatted_steps = format_steps(steps)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        problem=problem,
        steps=formatted_steps,
        num_steps=len(steps)
    )
    return SYSTEM_PROMPT, user_prompt
