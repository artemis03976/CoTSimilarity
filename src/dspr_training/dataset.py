import json
import random
import torch
from torch.utils.data import Dataset

from dspr.config import MATH_SYSTEM_PROMPT


def router_prompt_key(item):
    """Stable key used to join canonical prompts with cached contexts."""
    return (item['variant_type'], item['problem'])


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


class RouterPromptDataset(DSPRDataset):
    """Prompt-level router data from the canonical problem dataset."""

    def __init__(self, problem_dataset, tokenizer, max_length=2048, problem_ids=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = self._resolve_pad_token_id()
        wanted = None if problem_ids is None else {int(pid) for pid in problem_ids}
        self.data = []
        seen = set()
        for item in self._load_jsonl(problem_dataset):
            problem_id = item['problem_id']
            if wanted is not None and int(problem_id) not in wanted:
                continue
            for variant, target in (('simple', 0.0), ('hard', 1.0)):
                problem = item[variant]['problem'].strip()
                key = (variant, problem)
                if problem and key not in seen:
                    self.data.append({
                        'problem_id': problem_id,
                        'problem': problem,
                        'variant_type': variant,
                        'target_alpha': target,
                    })
                    seen.add(key)
        if not self.data:
            raise ValueError(f'Router dataset is empty: {problem_dataset}')
        self.contexts = None

    @classmethod
    def from_problem_dataset(cls, path, tokenizer, max_length=2048, problem_ids=None):
        return cls(path, tokenizer, max_length, problem_ids)

    def set_contexts(self, contexts):
        """Attach precomputed context vectors keyed by canonical prompt."""
        try:
            self.contexts = torch.stack(
                [contexts[router_prompt_key(item)] for item in self.data]
            )
        except KeyError as exc:
            raise ValueError(f'Missing cached context for router prompt: {exc.args[0]}') from exc

    def __len__(self):
        return len(self.data)

    @property
    def class_counts(self):
        simple = sum(float(item['target_alpha']) < 0.5 for item in self.data)
        return {'simple': int(simple), 'hard': len(self.data) - int(simple)}

    def __getitem__(self, idx):
        item = self.data[idx]
        output = {
            'target_alpha': torch.tensor([item['target_alpha']], dtype=torch.float32),
        }
        if self.contexts is not None:
            output['context_embeddings'] = self.contexts[idx]
            return output

        messages = [
            {'role': 'system', 'content': MATH_SYSTEM_PROMPT},
            {'role': 'user', 'content': item['problem']},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self._sanitize_token_ids(
            self.tokenizer.encode(prompt, add_special_tokens=False)
        )[:self.max_length]
        attention_mask = [1] * len(prompt_ids)
        padding_length = self.max_length - len(prompt_ids)
        prompt_ids += [self.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        output.update({
            'prompt_input_ids': torch.tensor(prompt_ids, dtype=torch.long),
            'prompt_attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        })
        return output


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
            variant = str(item.get('variant_type', item.get('variant', '')))
            problem = str(item.get('problem', ''))
            # A raw problem_id may be shared by simple/hard variants. Never
            # resample those distinct prompts as though they were one problem.
            group_key = (problem_id, variant, problem)
            self._groups.setdefault(group_key, []).append(index)
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
