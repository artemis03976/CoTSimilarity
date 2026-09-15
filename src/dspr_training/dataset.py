import json
import random
import torch
from torch.utils.data import Dataset

from dspr.config import MATH_SYSTEM_PROMPT


class DSPRDataset(Dataset):
    """Dataset for DSPR training."""

    def __init__(self, jsonl_path, tokenizer, max_length=2048):
        self.data = self._load_jsonl(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = self._resolve_pad_token_id()

    def _load_jsonl(self, path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data)

    def _resolve_pad_token_id(self):
        """Resolve a safe pad token id for tokenizers without explicit pad token."""
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        if self.tokenizer.unk_token_id is not None:
            return self.tokenizer.unk_token_id
        return 0

    def _sanitize_token_ids(self, token_ids):
        """Replace unexpected None token ids with a safe fallback id."""
        return [tid if tid is not None else self.pad_token_id for tid in token_ids]

    def __getitem__(self, idx):
        item = self.data[idx]

        return self._encode_item(item)

    def _encode_item(self, item):
        """Tokenize and format one trajectory record."""

        # Format prompt — must match inference-time format exactly
        messages = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": item['problem']},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize
        prompt_ids = self._sanitize_token_ids(
            self.tokenizer.encode(prompt, add_special_tokens=False)
        )
        response_ids = self._sanitize_token_ids(
            self.tokenizer.encode(item['response'], add_special_tokens=False)
        )

        # Build context-encoder input from prompt only
        prompt_input_ids = prompt_ids[:self.max_length]
        prompt_attention_mask = [1] * len(prompt_input_ids)
        prompt_padding_length = self.max_length - len(prompt_input_ids)
        prompt_input_ids += [self.pad_token_id] * prompt_padding_length
        prompt_attention_mask += [0] * prompt_padding_length

        # Combine and truncate for LM training
        input_ids = prompt_ids + response_ids
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        # Create labels (mask prompt tokens)
        labels = [-100] * len(prompt_ids) + response_ids
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]

        # Pad to max_length
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.pad_token_id] * padding_length
        labels += [-100] * padding_length
        attention_mask += [0] * padding_length

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'prompt_input_ids': torch.tensor(prompt_input_ids, dtype=torch.long),
            'prompt_attention_mask': torch.tensor(prompt_attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'target_alpha': torch.tensor([item['target_alpha']], dtype=torch.float32),
            'variant_type': item['variant_type']
        }


class ProblemResampledDSPRDataset(DSPRDataset):
    """One randomly selected trajectory per problem for each sampling epoch.

    The dataset length is the number of unique problems rather than the
    number of flattened trajectories. The training entrypoint compensates for
    this shorter sampling epoch with ``max_steps`` computed from the original
    flat dataset, preserving the optimizer update budget.
    """

    def __init__(self, jsonl_path, tokenizer, max_length=2048, seed=42):
        super().__init__(jsonl_path, tokenizer, max_length)
        self.seed = int(seed)
        self._groups = {}
        for index, item in enumerate(self.data):
            problem_id = str(item.get('problem_id'))
            self._groups.setdefault(problem_id, []).append(index)
        self._problem_ids = list(self._groups)
        if not self._problem_ids:
            raise ValueError('Cannot resample an empty DSPR dataset')
        self._sampled_indices = []
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
        """Choose exactly one trajectory per problem deterministically."""
        rng = random.Random(self.seed + int(epoch))
        self._sampled_indices = [
            rng.choice(self._groups[problem_id]) for problem_id in self._problem_ids
        ]

    def selected_indices(self):
        """Return sampled flat-record indices for diagnostics and tests."""
        return list(self._sampled_indices)

    def __getitem__(self, idx):
        return self._encode_item(self.data[self._sampled_indices[idx]])
