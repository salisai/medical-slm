"""Tests for dataset formatting utilities."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import build_user_content, example_to_messages


def test_build_user_content_instruction_and_input():
    row = {
        "instruction": "Answer the patient's question.",
        "input": "What are the symptoms of anemia?",
    }
    assert build_user_content(row) == (
        "Answer the patient's question.\n\nWhat are the symptoms of anemia?"
    )


def test_build_user_content_input_only():
    row = {"instruction": "", "input": "How is diabetes treated?"}
    assert build_user_content(row) == "How is diabetes treated?"


def test_example_to_messages():
    row = {
        "instruction": "",
        "input": "What causes headaches?",
        "output": "Headaches can be caused by stress, dehydration, and more.",
    }
    messages = example_to_messages(row)
    assert messages[0] == {"role": "user", "content": "What causes headaches?"}
    assert messages[1]["role"] == "assistant"
    assert "Headaches" in messages[1]["content"]
