"""QLoRA fine-tuning with SFTTrainer."""

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.config import LoRAConfig, ModelConfig, TrainingConfig


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_for_training(
    model_config: ModelConfig | None = None,
):
    model_config = model_config or ModelConfig()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_config.load_in_4bit,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, model_config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def build_lora_config(lora_config: LoRAConfig | None = None) -> LoraConfig:
    lora_config = lora_config or LoRAConfig()
    return LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
        target_modules=lora_config.target_modules,
    )


def build_trainer(
    train_dataset,
    training_config: TrainingConfig | None = None,
    model_config: ModelConfig | None = None,
    lora_config: LoRAConfig | None = None,
) -> SFTTrainer:
    training_config = training_config or TrainingConfig()
    model_config = model_config or ModelConfig()
    lora_config = lora_config or LoRAConfig()

    tokenizer = load_tokenizer(model_config.model_name)
    model = load_model_for_training(model_config)

    sft_config = SFTConfig(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_ratio=training_config.warmup_ratio,
        lr_scheduler_type=training_config.lr_scheduler_type,
        fp16=training_config.fp16,
        gradient_checkpointing=training_config.gradient_checkpointing,
        optim=training_config.optim,
        save_steps=training_config.save_steps,
        logging_steps=training_config.logging_steps,
        max_length=training_config.max_seq_length,
        packing=False,
        report_to="none",
        hub_model_id=training_config.hub_model_id,
        push_to_hub=training_config.hub_model_id is not None,
    )

    return SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(lora_config),
    )
