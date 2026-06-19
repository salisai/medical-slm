"""Training and inference configuration for QLoRA fine-tuning."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class DatasetConfig:
    dataset_name: str = "medalpaca/medical_meadow_mediqa"
    dataset_split: str = "train"
    max_samples: int | None = None  # None = use full dataset (~2.2k rows)


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )


@dataclass
class TrainingConfig:
    num_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    save_steps: int = 100
    logging_steps: int = 25
    output_dir: str = "./checkpoints"
    hub_model_id: str | None = None  # set to push adapter to Hugging Face Hub


@dataclass
class InferenceConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
