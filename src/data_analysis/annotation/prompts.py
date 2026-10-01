"""Prompt template for CoT dependency analysis."""

from typing import List, Dict, Tuple


SYSTEM_PROMPT = """Analyze a mathematical reasoning chain and construct its information dependency DAG.

[Input Format]
- [0] Original problem (Prompt)
- [1] to [N] are model-generated reasoning steps (already aggregated)

[Core Task]
For each step from 1 to N, strictly based on the literal text, determine which prerequisite information this step's derivation **directly uses**.
Annotate the given reasoning without solving the problem or correcting its mathematical errors.
Dependency sources may be combined:
1. Earlier step IDs: values, expressions, definitions, formulas, or conclusions directly used from those steps.
2. 0: conditions, definitions, or objectives taken directly from the original problem.
3. "External": newly introduced mathematical formulas, theorems, or unsupported numerical facts absent from the problem and earlier steps. Reusing a formula already stated earlier depends on that earlier step. Routine arithmetic and algebraic operations do not require "External".

[Macro-Action Classification]
For each step, you must also assign ONE macro-action tag from the following finite set:
- [Define]: Define variables or extract known conditions from the problem statement
- [Recall]: Recall or introduce external formulas/theorems (typically used with [External] dependency)
- [Derive]: Algebraic derivation, equation transformation, symbolic manipulation
- [Calculate]: Pure numerical calculation (arithmetic operations)
- [Verify]: Self-check, verification, or validation of previous results
- [Conclude]: Draw intermediate or final conclusions

Choose the PRIMARY action if a step involves multiple operations. Choose dependencies independently: Calculate or Derive may also use "External".

[Strict Rules]
1. Include all directly used sources. Exclude mere textual predecessors and indirect ancestors unless their information is also directly used.
2. Formula substitution depends on both the formula source and its inputs. If the formula is first introduced here, include "External" together with any input dependencies.
3. Keep analysis to one short sentence identifying the used information and its sources, consistent with depends_on.
4. Do not omit any step! Output exactly N objects in step order, each with exactly ONE tag.
5. Dependencies must be 0, "External", or earlier step IDs, without duplicates.

[One-shot Example]
Input: The problem gives radius r=5. Steps: (1) extracts r=5; (2) introduces A=pi*r^2; (3) substitutes r and uses pi=3.14 to calculate A.
Output:
[
  {
    "step_id": 1,
    "analysis": "Extracts r=5 from the problem.",
    "depends_on": [0],
    "macro_action_tag": "Define"
  },
  {
    "step_id": 2,
    "analysis": "Introduces the area formula A=pi*r^2.",
    "depends_on": ["External"],
    "macro_action_tag": "Recall"
  },
  {
    "step_id": 3, "analysis": "Uses r from step 1, the formula from step 2, and the new approximation pi=3.14.",
    "depends_on": [1, 2, "External"],
    "macro_action_tag": "Calculate"
  }
]

[Output Format]
Return only a valid JSON array with one object per input step."""


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
