"""Canonical prompt data for the DSPR router stage."""

import torch
from torch.utils.data import Dataset

from utils.datasets import PromptEncoder
from utils.io import read_jsonl


def router_prompt_key(item):
    """Stable key used to join canonical prompts with cached contexts."""
    return (item['variant_type'], item['problem'])


class RouterPromptDataset(Dataset):
    def __init__(self, problem_dataset, tokenizer, max_length=2048, problem_ids=None):
        self.encoder = PromptEncoder(tokenizer, max_length)
        wanted = None if problem_ids is None else {int(pid) for pid in problem_ids}
        self.data = []
        seen = set()
        for item in read_jsonl(problem_dataset):
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

    def __getitem__(self, index):
        item = self.data[index]
        output = {'target_alpha': torch.tensor([item['target_alpha']], dtype=torch.float32)}
        if self.contexts is not None:
            output['context_embeddings'] = self.contexts[index]
        else:
            prompt_ids, prompt_mask = self.encoder.pad(self.encoder.prompt_ids(item['problem']))
            output.update(prompt_input_ids=prompt_ids, prompt_attention_mask=prompt_mask)
        return output
