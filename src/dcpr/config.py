from dataclasses import dataclass

MATH_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


@dataclass
class DCPRConfig:
    """Configuration for DCPR model."""
    # Model architecture
    model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    context_layer_idx: int = 15
    prefix_length: int = 50
    router_intermediate_dim: int = 256
    router_dropout: float = 0.05

    # Training
    learning_rate: float = 4e-5
    batch_size: int = 4
    num_epochs: int = 15
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # Loss weights
    lambda_router: float = 0.5

    # Data
    train_data_path: str = "data/qwen/dcpr_train.jsonl"
    val_data_path: str = "data/qwen/dcpr_val.jsonl"
    checkpoint_dir: str = "checkpoints"
    max_seq_length: int = 2048

    # Reproducibility
    seed: int = 42

    # Hardware
    device: str = "cuda"
    gradient_checkpointing: bool = True
