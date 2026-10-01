"""Shared identities and the existing JSON annotation fields."""

import re
from dataclasses import dataclass
from typing import TypedDict


VARIANTS = ("original", "simple", "hard")
VALID_TAGS = {"Define", "Recall", "Derive", "Calculate", "Verify", "Conclude"}


class Step(TypedDict):
    index: int
    text: str


class DAGNode(TypedDict):
    step_id: int
    depends_on: list[int | str]
    macro_action_tag: str


@dataclass(frozen=True)
class SampleKey:
    problem_id: int
    variant: str
    sample_index: int = 0

    def __str__(self) -> str:
        return f"{self.problem_id}_{self.variant}_{self.sample_index}"

    @classmethod
    def parse(cls, value: str) -> "SampleKey":
        match = re.fullmatch(r"(\d+)_(original|simple|hard)(?:_(\d+))?", value)
        if match is None:
            raise ValueError(f"Invalid sample ID: {value}")
        return cls(int(match[1]), match[2], int(match[3] or 0))
