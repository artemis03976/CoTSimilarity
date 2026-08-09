"""Unified deterministic generation for Base, DSPR, SPT, and LoRA models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .common import GenerationResult, build_prompt


BASELINES = ("base", "dspr", "spt", "lora")


@dataclass(frozen=True)
class GreedyModelConfig:
    baseline: str
    model_name: str
    checkpoint: Path | None = None
    device: str = "cuda"
    dtype: str = "float16"
    attention_implementation: str = "flash_attention_2"
    max_new_tokens: int = 4096
    system_prompt: str | None = None
    context_layer_idx: int = 15
    prefix_length: int = 50
    router_intermediate_dim: int = 256
    router_dropout: float = 0.05
    max_seq_length: int = 4096
    forced_alpha: float | None = None

    def __post_init__(self) -> None:
        if self.baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}, got {self.baseline!r}")
        if self.baseline != "base" and self.checkpoint is None:
            raise ValueError(f"checkpoint is required for baseline={self.baseline}")
        if self.max_new_tokens < 1 or self.max_seq_length < 1:
            raise ValueError("token limits must be positive")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.forced_alpha is not None:
            if self.baseline != "dspr":
                raise ValueError("forced_alpha is only valid for DSPR")
            if not math.isfinite(self.forced_alpha) or not 0.0 <= self.forced_alpha <= 1.0:
                raise ValueError("forced_alpha must be finite and in [0, 1]")


class GreedyGenerator(Protocol):
    """Minimal interface consumed by the deterministic experiment runner."""

    baseline: str

    def generate(self, problem: str) -> GenerationResult:
        ...


def _torch_dtype(torch: Any, name: str) -> Any:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _pad_token_id(tokenizer: Any) -> int | None:
    return (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )


def _eos_token_ids(tokenizer: Any) -> set[int]:
    value = tokenizer.eos_token_id
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(token) for token in value}


def _finish_reason(token_ids: tuple[int, ...], tokenizer: Any, max_new_tokens: int) -> str:
    eos_ids = _eos_token_ids(tokenizer)
    if token_ids and token_ids[-1] in eos_ids:
        return "stop"
    return "length" if len(token_ids) >= max_new_tokens else "stop"


class _TorchGeneratorBase:
    baseline: str

    def __init__(self, tokenizer: Any, device: str, max_new_tokens: int, system_prompt: str | None):
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt

    def _inputs(self, problem: str) -> tuple[Any, Any]:
        import torch

        prompt = (
            build_prompt(self.tokenizer, problem)
            if self.system_prompt is None
            else build_prompt(self.tokenizer, problem, self.system_prompt)
        )
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        return input_ids, torch.ones_like(input_ids)

    def _generation_kwargs(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": _pad_token_id(self.tokenizer),
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def _prefix_model_generation_kwargs(self) -> dict[str, Any]:
        """DSPR/SPT wrappers already supply pad and EOS IDs internally."""

        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }

    def _result(
        self,
        new_token_ids: Any,
        metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        ids = tuple(int(token) for token in new_token_ids[0].detach().cpu().tolist())
        text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            token_ids=ids,
            finish_reason=_finish_reason(ids, self.tokenizer, self.max_new_tokens),
            metadata=metadata or {},
        )


class TransformersGreedyGenerator(_TorchGeneratorBase):
    """Greedy generator for a base model or PEFT-wrapped LoRA model."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        baseline: str,
        device: str,
        max_new_tokens: int,
        system_prompt: str | None,
    ) -> None:
        super().__init__(tokenizer, device, max_new_tokens, system_prompt)
        self.model = model
        self.baseline = baseline

    def generate(self, problem: str) -> GenerationResult:
        import torch

        input_ids, attention_mask = self._inputs(problem)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self._generation_kwargs(),
            )
        continuation = output_ids[:, input_ids.shape[1] :]
        return self._result(continuation)


class SPTGreedyGenerator(_TorchGeneratorBase):
    baseline = "spt"

    def __init__(self, model: Any, max_new_tokens: int, system_prompt: str | None) -> None:
        super().__init__(model.tokenizer, model.config.device, max_new_tokens, system_prompt)
        self.model = model

    def generate(self, problem: str) -> GenerationResult:
        import torch

        input_ids, attention_mask = self._inputs(problem)
        with torch.inference_mode():
            continuation = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self._prefix_model_generation_kwargs(),
            )
        return self._result(continuation)


class DSPRGreedyGenerator(_TorchGeneratorBase):
    baseline = "dspr"

    def __init__(
        self,
        model: Any,
        max_new_tokens: int,
        system_prompt: str | None,
        forced_alpha: float | None,
    ) -> None:
        super().__init__(model.tokenizer, model.config.device, max_new_tokens, system_prompt)
        self.model = model
        self.forced_alpha = forced_alpha

    def generate(self, problem: str) -> GenerationResult:
        import torch

        input_ids, attention_mask = self._inputs(problem)
        with torch.inference_mode():
            continuation, alpha = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_input_ids=input_ids.clone(),
                prompt_attention_mask=attention_mask.clone(),
                forced_alpha=self.forced_alpha,
                **self._prefix_model_generation_kwargs(),
            )
        return self._result(continuation, {"alpha": float(alpha.item())})


def _model_load_kwargs(config: GreedyModelConfig, torch: Any) -> dict[str, Any]:
    attention = config.attention_implementation
    if config.device.startswith("cpu") and attention == "flash_attention_2":
        attention = "eager"
    return {
        "torch_dtype": _torch_dtype(torch, config.dtype),
        "trust_remote_code": True,
        "attn_implementation": attention,
    }


def load_greedy_generator(config: GreedyModelConfig) -> GreedyGenerator:
    """Load the requested model family behind one deterministic interface."""

    import torch

    if config.baseline in {"base", "lora"}:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **_model_load_kwargs(config, torch),
        )
        if config.baseline == "lora":
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("LoRA inference requires the peft package") from exc
            model = PeftModel.from_pretrained(
                model,
                str(config.checkpoint),
                is_trainable=False,
            )
        model.config.use_cache = True
        model.to(config.device)
        model.eval()
        return TransformersGreedyGenerator(
            model=model,
            tokenizer=tokenizer,
            baseline=config.baseline,
            device=config.device,
            max_new_tokens=config.max_new_tokens,
            system_prompt=config.system_prompt,
        )

    if config.baseline == "dspr":
        from dspr.config import DSPRConfig
        from dspr.model import DSPRModel

        model_config = DSPRConfig(
            model_name=config.model_name,
            context_layer_idx=config.context_layer_idx,
            prefix_length=config.prefix_length,
            router_intermediate_dim=config.router_intermediate_dim,
            router_dropout=config.router_dropout,
            max_seq_length=config.max_seq_length,
            device=config.device,
            gradient_checkpointing=False,
            checkpoint_dir=str(config.checkpoint.parent),
        )
        model = DSPRModel(model_config)
        checkpoint = torch.load(config.checkpoint, map_location=config.device)
        model.dual_prefix.load_state_dict(checkpoint["dual_prefix_state_dict"])
        model.router.load_state_dict(checkpoint["router_state_dict"])
        model.to(config.device)
        model.eval()
        return DSPRGreedyGenerator(
            model,
            config.max_new_tokens,
            config.system_prompt,
            config.forced_alpha,
        )

    from spt import SPTConfig, StaticPromptModel

    model_config = SPTConfig(
        model_name=config.model_name,
        prefix_length=config.prefix_length,
        max_seq_length=config.max_seq_length,
        device=config.device,
        gradient_checkpointing=False,
    )
    model = StaticPromptModel(model_config)
    checkpoint = torch.load(config.checkpoint, map_location=config.device)
    model.soft_prompt.data.copy_(checkpoint["soft_prompt"])
    model.to(config.device)
    model.eval()
    return SPTGreedyGenerator(model, config.max_new_tokens, config.system_prompt)
