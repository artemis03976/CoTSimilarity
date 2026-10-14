"""Single-GPU, batched base inference with the shared greedy evaluation protocol."""

from __future__ import annotations

from importlib.metadata import version
from typing import Sequence

from .common import GenerationResult, build_prompt
from .greedy import GreedyModelConfig, _eos_token_ids


class VLLMGreedyGenerator:
    baseline = "base"

    def __init__(
        self,
        config: GreedyModelConfig,
        *,
        problems: Sequence[str] | None = None,
    ) -> None:
        from transformers import AutoConfig, AutoTokenizer, GenerationConfig

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        self._prompt_ids: dict[str, list[int]] = {}
        for problem in problems or ():
            self._encode_problem(problem)
        max_prompt_tokens = max((len(ids) for ids in self._prompt_ids.values()), default=0)
        required_length = max_prompt_tokens + config.max_new_tokens
        self.max_model_len = config.max_model_len or max(4096, required_length)
        if required_length > self.max_model_len:
            raise ValueError(
                f"Longest prompt ({max_prompt_tokens} tokens) + max_new_tokens "
                f"({config.max_new_tokens}) exceeds max_model_len={self.max_model_len}; "
                f"set --max-model-len to at least {required_length}. Inputs are not truncated."
            )
        if not config.enable_chunked_prefill and config.max_num_batched_tokens < self.max_model_len:
            raise ValueError("Without chunked prefill, max_num_batched_tokens must cover max_model_len")

        # HF generate inherits repetition/minimum-length settings from this
        # configuration, but our existing adapter explicitly overrides its EOS.
        try:
            generation = GenerationConfig.from_pretrained(config.model_name)
        except OSError:
            generation = GenerationConfig.from_model_config(
                AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
            )
        if generation.num_beams != 1 or generation.num_return_sequences != 1:
            raise ValueError("vLLM greedy requires a single-sequence, non-beam model generation config")
        unsupported = {
            "no_repeat_ngram_size": 0,
            "encoder_repetition_penalty": 1.0,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "bad_words_ids": None,
            "suppress_tokens": None,
            "begin_suppress_tokens": None,
            "stop_strings": None,
            "sequence_bias": None,
        }
        for name, default in unsupported.items():
            if getattr(generation, name, default) != default:
                raise ValueError(
                    f"Model generation setting {name} requires --backend transformers "
                    "to preserve its decoding behavior"
                )
        self._min_new_tokens = generation.min_new_tokens
        self._min_length = generation.min_length
        self._sampling_kwargs = {
            "n": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": generation.repetition_penalty,
            "max_tokens": config.max_new_tokens,
            "seed": config.seed,
            "stop_token_ids": sorted(_eos_token_ids(self.tokenizer)),
            # Stop only on the tokenizer EOS, matching the existing HF adapter.
            # This also prevents model generation_config from adding extra EOS.
            "ignore_eos": True,
            "skip_special_tokens": True,
        }
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("The vLLM greedy backend requires the vllm package") from exc
        self.sampling_params_class = SamplingParams
        print(
            f"vLLM: {len(problems or ())} prompts, longest={max_prompt_tokens} tokens, "
            f"max_model_len={self.max_model_len}, max_num_seqs={config.max_num_seqs}"
        )
        self.llm = LLM(
            model=config.model_name,
            trust_remote_code=True,
            dtype=config.dtype,
            tensor_parallel_size=1,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            seed=config.seed,
        )
        self._closed = False
        self.run_metadata = {
            "backend": "vllm",
            "vllm_version": version("vllm"),
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_prompt_tokens": max_prompt_tokens,
            "max_num_seqs": config.max_num_seqs,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "enable_chunked_prefill": config.enable_chunked_prefill,
            "sampling": dict(self._sampling_kwargs),
            "min_new_tokens": self._min_new_tokens,
            "min_length": self._min_length,
        }

    def _encode_problem(self, problem: str) -> list[int]:
        if problem not in self._prompt_ids:
            prompt = (
                build_prompt(self.tokenizer, problem)
                if self.config.system_prompt is None
                else build_prompt(self.tokenizer, problem, self.config.system_prompt)
            )
            self._prompt_ids[problem] = self.tokenizer.encode(prompt, add_special_tokens=False)
        return self._prompt_ids[problem]

    def generate(self, problem: str) -> GenerationResult:
        return self.generate_batch([problem])[0]

    def generate_batch(self, problems: Sequence[str]) -> list[GenerationResult]:
        prompts = []
        params = []
        for problem in problems:
            ids = self._encode_problem(problem)
            if len(ids) + self.config.max_new_tokens > self.max_model_len:
                raise ValueError("Prompt plus output budget exceeds max_model_len; increase --max-model-len")
            prompts.append({"prompt_token_ids": ids})
            min_tokens = (
                self._min_new_tokens
                if self._min_new_tokens is not None
                else max(0, self._min_length - len(ids))
            )
            params.append(self.sampling_params_class(**self._sampling_kwargs, min_tokens=min_tokens))
        outputs = self.llm.generate(prompts, sampling_params=params, use_tqdm=True)
        if len(outputs) != len(problems):
            raise RuntimeError("vLLM did not return one output per input prompt")
        results = []
        for prompt, output in zip(prompts, outputs):
            if list(output.prompt_token_ids) != prompt["prompt_token_ids"]:
                raise RuntimeError("vLLM output order does not match the submitted prompts")
            if len(output.outputs) != 1:
                raise RuntimeError("Expected exactly one greedy completion per prompt")
            completion = output.outputs[0]
            token_ids = tuple(int(token) for token in completion.token_ids)
            results.append(
                GenerationResult(
                    text=self.tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
                    token_ids=token_ids,
                    finish_reason=completion.finish_reason,
                )
            )
        return results

    def close(self) -> None:
        """Release the offline engine, including V0/V1 worker processes."""
        if self._closed:
            return
        self._closed = True
        engine = getattr(self.llm, "llm_engine", None)
        for target in (self.llm, engine, getattr(engine, "model_executor", None)):
            shutdown = getattr(target, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass
                break
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
        except Exception:
            pass
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
