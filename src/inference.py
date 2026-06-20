"""Run inference with a base model or a saved QLoRA adapter."""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import InferenceConfig, ModelConfig


def load_model_for_inference(
    model_name: str | None = None,
    adapter_path: str | None = None,
    load_in_4bit: bool = True,
):
    """Load tokenizer and model (optionally with a LoRA adapter)."""
    model_config = ModelConfig()
    model_name = model_name or model_config.model_name

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(
                torch, model_config.bnb_4bit_compute_dtype
            ),
            bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return tokenizer, model


def generate_response(
    tokenizer,
    model,
    user_message: str,
    system_message: str | None = None,
    config: InferenceConfig | None = None,
) -> str:
    """Generate an assistant reply for a user message."""
    config = config or InferenceConfig()

    messages: list[dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
