# Medical-SLM

Fine-tune **SmolLM2-360M-Instruct** on medical patient Q&A using **QLoRA** .

Teach a small instruction-tuned language model to answer patient health questions using the [Medical Meadow MediQA](https://huggingface.co/datasets/medalpaca/medical_meadow_mediqa) dataset.

> **Disclaimer:** This project is for learning and research. The fine-tuned model is **not** a substitute for professional medical advice, diagnosis, or treatment.

## Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Model | [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 360M params, chat-ready |
| Dataset | [medalpaca/medical_meadow_mediqa](https://huggingface.co/datasets/medalpaca/medical_meadow_mediqa) | Real patient questions + expert answers (~2.2k rows) |
| Method | QLoRA (4-bit + LoRA) | ~3–5 GB VRAM, close to full fine-tuning quality |

## Project structure

```
medical-slm/
├── notebooks/
│   └── finetune.ipynb      # Main Colab notebook — run this to train
├── src/
│   ├── config.py           # Model, LoRA, and training settings
│   ├── data_utils.py       # Dataset loading and chat formatting
│   ├── train_utils.py      # QLoRA + SFTTrainer setup
│   └── inference.py        # Load adapter and generate replies
├── tests/
│   └── test_data_utils.py
├── requirements.txt
├── .env.example
└── README.md
```

## Quick start (Google Colab)

1. Open [`notebooks/finetune.ipynb`](notebooks/finetune.ipynb) in Colab.
2. **Runtime → Change runtime type → T4 GPU**
3. Update the clone URL in the first code cell with your GitHub repo (or upload the project folder).
4. Run all cells.

Training saves LoRA adapters to `./checkpoints` (or Google Drive if enabled).

### Expected runtime

| Setting | Value |
|---------|-------|
| Epochs | 2 |
| Batch size | 2 (effective 8 with grad accumulation) |
| Max sequence length | 512 |
| GPU | Colab T4 |
| Time | ~20–40 minutes |

## Local setup (optional)

```bash
git clone https://github.com/salisai/medical-slm.git
cd medical-slm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Local training requires a CUDA GPU with ~6 GB+ VRAM.

## Configuration

All defaults live in [`src/config.py`](src/config.py):

```python
from src.config import ModelConfig, LoRAConfig, TrainingConfig, DatasetConfig

model_config = ModelConfig()          # SmolLM2-360M-Instruct, 4-bit
lora_config = LoRAConfig()            # r=16, alpha=32, q_proj + v_proj
training_config = TrainingConfig()    # 2 epochs, batch 2, max_len 512
dataset_config = DatasetConfig()      # medical_meadow_mediqa
```

Override any field in the notebook or your own script.

## Push to Hugging Face Hub (optional)

1. Copy `.env.example` → `.env` and set `HF_TOKEN`.
2. In the notebook, uncomment `login(...)` and set `hub_model_id`:

```python
training_config = TrainingConfig(
    output_dir=CHECKPOINT_DIR,
    hub_model_id="your-username/smol-medical-coach",
)
```

## Inference after training

```python
from src.inference import load_model_for_inference, generate_response

tokenizer, model = load_model_for_inference(
    model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
    adapter_path="./checkpoints",
)

answer = generate_response(
    tokenizer,
    model,
    "What are common symptoms of iron deficiency anemia?",
)
print(answer)
```

## Contributing

Issues and pull requests welcome. This is a learning project — keep changes focused and well documented.
