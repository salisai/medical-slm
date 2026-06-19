"""Load and format the medical Q&A dataset for chat-style SFT."""

from typing import Any

from datasets import Dataset, load_dataset

from src.config import DatasetConfig


def build_user_content(example: dict[str, Any]) -> str:
    """Combine Alpaca-style instruction and input into a single user message."""
    instruction = (example.get("instruction") or "").strip()
    user_input = (example.get("input") or "").strip()

    if instruction and user_input:
        return f"{instruction}\n\n{user_input}"
    return instruction or user_input


def example_to_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    """Convert one dataset row into chat messages."""
    return [
        {"role": "user", "content": build_user_content(example)},
        {"role": "assistant", "content": (example.get("output") or "").strip()},
    ]


def load_medical_dataset(config: DatasetConfig | None = None) -> Dataset:
    """Load medalpaca/medical_meadow_mediqa and format it for SFTTrainer."""
    config = config or DatasetConfig()

    dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    if config.max_samples is not None:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))

    dataset = dataset.map(
        lambda row: {"messages": example_to_messages(row)},
        remove_columns=dataset.column_names,
        desc="Formatting chat messages",
    )

    return dataset
