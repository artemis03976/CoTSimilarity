import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from utils import is_target_module, map_name
from soft_prompt.layer_map import init_keywords_map


class SoftPromptGenerator(nn.Module):
    def __init__(
        self, 
        base_model,
        len_soft_prompt=None, 
        num_layers=None, 
        embed_idx=None,
        dynamic=True
    ):
        super(SoftPromptGenerator, self).__init__()

        self.model_name = base_model.model_name
        self.len_soft_prompt = len_soft_prompt
        self.num_layers = num_layers
        self.embed_idx = embed_idx
        self.dynamic = dynamic

        if self.dynamic:
            generator_config = AutoConfig.from_pretrained(self.model_name)
            generator_config.vocab_size = base_model.config.vocab_size

            if 'gpt' in self.model_name:
                generator_config.num_layers = self.num_layers
                generator_config.attention_layers = generator_config.attention_layers[:self.num_layers]

                self.generator = self.build_generator(base_model.transformer, generator_config)
            
            elif 'pythia' in self.model_name:
                generator_config.num_hidden_layers = self.num_layers
                self.generator = self.build_generator(base_model.gpt_neox, generator_config)
        else:
            embedding = base_model.get_input_embeddings().weight[:self.len_soft_prompt].clone().detach()
            self.generator = nn.Parameter(embedding)

    def build_generator(self, base_model, generator_config):
        new_generator = AutoModel.from_config(generator_config)
        new_generator.load_state_dict(base_model.state_dict(), strict=False)

        # zero out related parameters
        keyword_list = map_name(init_keywords_map, self.model_name)
        for module_name, module in new_generator.named_modules():
            if is_target_module(keyword_list, module_name):
                for name, val in module.named_parameters():
                    setattr(module, name, nn.Parameter(torch.zeros_like(val)))

        if 'gpt' in self.model_name:
            new_generator.wpe.weight = nn.Parameter(torch.zeros_like(new_generator.wpe.weight))

        return new_generator

    def forward(self, input_ids):
        if self.dynamic:
            soft_embedding = self.generator(
                input_ids=input_ids[:, -self.len_soft_prompt:], 
                output_hidden_states=True
            )['hidden_states'][self.embed_idx]
        else:
            soft_embedding = self.generator.repeat(input_ids.size(0), 1, 1)

        return soft_embedding
    
    def save(self, save_path):
        if isinstance(self.generator, nn.Parameter):
            state_dict = self.generator.data
            bf16_state_dict = state_dict.to(torch.bfloat16)
        else:
            state_dict = self.generator.state_dict()
            bf16_state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
        
        torch.save(bf16_state_dict, save_path)
    
    def load(self, load_path):
        sft_state_dict = torch.load(load_path, weights_only=True)

        if isinstance(self.generator, nn.Parameter):
            self.generator.data = sft_state_dict
        else:
            sft_embedding_weights = sft_state_dict.pop('wte.weight', None)
            self.generator.load_state_dict(sft_state_dict, strict=False)
            
            if sft_embedding_weights is not None:
                current_embedding = self.generator.get_input_embeddings()
                
                with torch.no_grad():
                    num_sft_tokens = sft_embedding_weights.shape[0]
                    current_embedding.weight[:num_sft_tokens, :] = sft_embedding_weights


class SoftEmbedding(nn.Module):
    def __init__(self, generator, wte: nn.Embedding, len_prompt):
        super(SoftEmbedding, self).__init__()
        
        self.generator = generator
        self.wte = wte
        self.len_prompt = len_prompt

    def forward(self, input_ids):
        prompt = input_ids[:, :self.len_prompt]
        target = input_ids[:, self.len_prompt:]
        
        if self.generator.dynamic:
            soft_prompts = self.generator(input_ids=prompt)
            prompt_embedding = self.wte(prompt[:, :-soft_prompts.shape[1]])
            target_embedding = self.wte(target)

            embedding = torch.cat([prompt_embedding, soft_prompts, target_embedding], dim=1)
        else:
            soft_prompts = self.generator(input_ids=prompt)
            prompt_embedding = self.wte(prompt)
            target_embedding = self.wte(target)

            embedding = torch.cat([soft_prompts, prompt_embedding, target_embedding], dim=1)

        return embedding

    def save(self, save_path):
        self.generator.save(save_path)

    def load(self, load_path):
        self.generator.load(load_path)