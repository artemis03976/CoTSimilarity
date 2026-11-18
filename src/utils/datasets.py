"""Trajectory tokenization and per-problem sampling shared by all training methods."""

import random

import torch
from torch.utils.data import Dataset

from .io import read_records
from .prompts import build_prompt


class PromptEncoder:
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError('Tokenizer needs a pad or EOS token for batched training')

    def token_ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def prompt_ids(self, problem):
        return self.token_ids(build_prompt(self.tokenizer, problem))

    def pad(self, token_ids):
        ids = token_ids[:self.max_length]
        padding = self.max_length - len(ids)
        mask = [1] * len(ids) + [0] * padding
        ids += [self.pad_token_id] * padding
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


class TrajectoryDataset(Dataset):
    """Mask prompt tokens in the LM objective; optionally supply router inputs."""

    def __init__(self, jsonl_path, tokenizer, max_length=2048, *, include_router_inputs=False):
        self.data = read_records(jsonl_path)
        self.encoder = PromptEncoder(tokenizer, max_length)
        self.include_router_inputs = include_router_inputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self._encode_item(self.data[index])

    def _encode_item(self, item):
        prompt_ids = self.encoder.prompt_ids(item['problem'])
        response_ids = self.encoder.token_ids(item['response'])
        input_ids, attention_mask = self.encoder.pad(prompt_ids + response_ids)
        labels = ([-100] * len(prompt_ids) + response_ids)[:self.encoder.max_length]
        labels += [-100] * (self.encoder.max_length - len(labels))
        output = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels, dtype=torch.long),
        }
        if self.include_router_inputs:
            prompt_input_ids, prompt_attention_mask = self.encoder.pad(prompt_ids)
            output.update(
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                target_alpha=torch.tensor([item['target_alpha']], dtype=torch.float32),
                variant_type=item['variant_type'],
            )
            if 'ged_weight' in item:
                output['ged_weight'] = torch.tensor(item['ged_weight'], dtype=torch.float32)
        return output


class ProblemResampledDataset(TrajectoryDataset):
    """One trajectory per distinct problem/variant/prompt at each sampling epoch.

    Callers retain the optimizer update budget computed from the flat dataset.
    """

    def __init__(self, jsonl_path, tokenizer, max_length=2048, seed=42,
                 *, include_router_inputs=False):
        super().__init__(jsonl_path, tokenizer, max_length,
                         include_router_inputs=include_router_inputs)
        self.seed = int(seed)
        self._groups = {}
        for index, item in enumerate(self.data):
            key = (str(item.get('problem_id')),
                   str(item.get('variant_type', item.get('variant', ''))),
                   str(item.get('problem', '')))
            self._groups.setdefault(key, []).append(index)
        self._problem_ids = list(self._groups)
        if not self._problem_ids:
            raise ValueError('Cannot resample an empty trajectory dataset')
        self.set_epoch(0)

    @property
    def num_problems(self):
        return len(self._problem_ids)

    @property
    def num_trajectories(self):
        return len(self.data)

    def __len__(self):
        return self.num_problems

    def set_epoch(self, epoch: int):
        rng = random.Random(self.seed + int(epoch))
        self._sampled_indices = [
            rng.choice(self._groups[problem_id]) for problem_id in self._problem_ids
        ]

    def __getitem__(self, index):
        return self._encode_item(self.data[self._sampled_indices[index]])
