# Apertus 8B Post-Training Framework

A comprehensive framework for post-training the Apertus 8B parameter model using various methods including SFT, DPO, and ORPO. Optimized for single GPU training with QLoRA/LoRA.

## Table of Contents

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Post-Training Methods](#post-training-methods)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference and Testing](#inference-and-testing)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Overview

This framework provides everything you need to perform post-training on the Apertus 8B model:

- **Multiple Methods**: SFT, DPO, ORPO, and more
- **Memory Efficient**: QLoRA support for 10-16GB GPUs
- **Complete Pipeline**: From data prep to inference
- **Well Documented**: Examples and templates included
- **Flexible**: Configurable for different hardware setups

### What is Post-Training?

Post-training (or fine-tuning) adapts a pre-trained base model to:
- Follow specific instructions better
- Align with human preferences
- Perform domain-specific tasks
- Exhibit desired behaviors and styles

---

## Hardware Requirements

### GPU Training (Recommended)

| Method | VRAM Needed | GPU Examples | Technique |
|--------|-------------|--------------|-----------|
| SFT | 10-12GB | RTX 3090, 4080, T4 | QLoRA 4-bit |
| SFT | 16-24GB | RTX 4090, A10G, L4 | LoRA 16-bit |
| SFT | 40-80GB | A100, H100 | Full fine-tuning |
| DPO | 12-16GB | RTX 4090, A10G | QLoRA 4-bit |
| ORPO | 12-16GB | RTX 4090, A10G | QLoRA 4-bit |

### CPU Training (Not Recommended)

**Feasible but extremely slow:**
- Requires 64GB+ RAM
- 50-100x slower than GPU
- Only practical for testing or tiny datasets (< 100 examples)

**Recommendation**: Use cloud GPU providers (Vast.ai, RunPod, Lambda Labs) if you don't have local GPU access.

---

## Installation

### 1. Clone/Setup Project

```bash
cd /path/to/your/project
```

### 2. Create Virtual Environment

```bash
# Using venv (already created in your case)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
# conda create -n apertus-training python=3.10
# conda activate apertus-training
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: If `flash-attn` fails to install (requires CUDA), comment it out in `requirements.txt` or install without it:

```bash
pip install -r requirements.txt --no-deps flash-attn || true
```

### 4. Login to Hugging Face (for model access)

```bash
pip install huggingface-hub
huggingface-cli login
```

### 5. (Optional) Setup Weights & Biases

```bash
pip install wandb
wandb login
```

---

## Project Structure

```
.
├── datasets/              # Training datasets
│   ├── sft_template.jsonl
│   ├── dpo_template.jsonl
│   ├── kto_template.jsonl
│   └── README_datasets.md
├── scripts/               # Training scripts
│   ├── train_sft.py
│   ├── train_dpo.py
│   └── train_orpo.py
├── configs/               # Configuration files
│   ├── sft_qlora_single_gpu.sh
│   ├── dpo_qlora.sh
│   └── orpo_qlora.sh
├── utils/                 # Utility scripts
│   ├── data_prep.py
│   └── inference.py
├── outputs/               # Trained models and checkpoints
├── models/                # Downloaded base models (optional)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

---

## Quick Start

### 1. Prepare Your Dataset

Use the provided templates or create your own:

```bash
# Copy and edit the SFT template
cp datasets/sft_template.jsonl datasets/my_data.jsonl
# Edit my_data.jsonl with your examples
```

### 2. Run Training

```bash
# Make config executable
chmod +x configs/sft_qlora_single_gpu.sh

# Start training
bash configs/sft_qlora_single_gpu.sh
```

### 3. Test Your Model

```python
from utils.inference import InferenceEngine

engine = InferenceEngine("outputs/sft_qlora_8b")
response = engine.chat("Explain quantum computing in simple terms.")
print(response)
```

---

## Post-Training Methods

### 1. Supervised Fine-Tuning (SFT)

**What**: Train model on instruction-response pairs
**When**: First stage, teaching model to follow instructions
**Data Format**: `{instruction, input, output}`

```bash
python scripts/train_sft.py \
    --model_name "apertus/apertus-8b" \
    --dataset_path "datasets/sft_template.jsonl" \
    --output_dir "outputs/my_sft_model"
```

### 2. Direct Preference Optimization (DPO)

**What**: Align model using preference pairs (chosen vs rejected)
**When**: After SFT, to align model preferences
**Data Format**: `{prompt, chosen, rejected}`

```bash
python scripts/train_dpo.py \
    --model_name "outputs/my_sft_model" \
    --dataset_path "datasets/dpo_template.jsonl" \
    --output_dir "outputs/my_dpo_model"
```

### 3. Odds Ratio Preference Optimization (ORPO)

**What**: Combines SFT + preference learning in one stage
**When**: Alternative to SFT→DPO pipeline, more efficient
**Data Format**: Same as DPO: `{prompt, chosen, rejected}`

```bash
python scripts/train_orpo.py \
    --model_name "apertus/apertus-8b" \
    --dataset_path "datasets/dpo_template.jsonl" \
    --output_dir "outputs/my_orpo_model"
```

### Comparison

| Method | Stages | Speed | Quality | Use Case |
|--------|--------|-------|---------|----------|
| SFT | 1 | Fast | Good | Instruction following |
| SFT→DPO | 2 | Moderate | Better | Aligned responses |
| ORPO | 1 | Fast | Better | Efficient alignment |

---

## Dataset Preparation

### Format Requirements

#### SFT Dataset
```json
{"instruction": "Question or task", "input": "Optional context", "output": "Desired response"}
```

#### DPO/ORPO Dataset
```json
{"prompt": "Question", "chosen": "Good response", "rejected": "Bad response"}
```

#### KTO Dataset
```json
{"prompt": "Question", "completion": "A response", "label": true}
```

### Creating Datasets

#### Method 1: Manual Creation

Edit the templates in `datasets/` directory.

#### Method 2: Convert from CSV

```python
from utils.data_prep import convert_csv_to_sft

convert_csv_to_sft(
    csv_path="your_data.csv",
    output_path="datasets/my_sft_data.jsonl",
    instruction_col="question",
    output_col="answer"
)
```

#### Method 3: Generate with LLMs

```python
# Use OpenAI, Anthropic, or other APIs to generate training data
# Example: Generate diverse Q&A pairs for your domain
```

### Data Validation

```bash
python utils/data_prep.py validate-sft --input datasets/my_data.jsonl
python utils/data_prep.py validate-dpo --input datasets/my_dpo_data.jsonl
```

### Data Analysis

```bash
python utils/data_prep.py analyze --input datasets/my_data.jsonl
```

### Data Splitting

```bash
python utils/data_prep.py split \
    --input datasets/my_data.jsonl \
    --output datasets/train.jsonl \
    --test-size 0.1
```

---

## Training

### Configuration Options

All training scripts accept these key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | Base model to train | apertus/apertus-8b |
| `--dataset_path` | Training data path | datasets/*.jsonl |
| `--output_dir` | Where to save model | outputs/* |
| `--use_lora` | Use LoRA | true |
| `--lora_r` | LoRA rank | 16 |
| `--use_4bit` | Use 4-bit quantization | true |
| `--num_train_epochs` | Training epochs | 3 (SFT), 1 (DPO) |
| `--per_device_train_batch_size` | Batch size | 4 (SFT), 2 (DPO) |
| `--learning_rate` | Learning rate | 2e-4 (SFT), 5e-5 (DPO) |
| `--max_seq_length` | Max sequence length | 2048 |

### Memory Optimization Tips

**If you run out of memory:**

1. **Reduce batch size**: `--per_device_train_batch_size 1`
2. **Increase gradient accumulation**: `--gradient_accumulation_steps 16`
3. **Reduce sequence length**: `--max_seq_length 1024`
4. **Use smaller LoRA rank**: `--lora_r 8`
5. **Enable gradient checkpointing**: (automatic with QLoRA)

### Monitoring Training

#### Using Weights & Biases

```bash
# Training will automatically log to W&B if enabled
--use_wandb true --wandb_project "my-project"
```

#### Using TensorBoard

```bash
tensorboard --logdir outputs/sft_qlora_8b/runs
```

#### Monitor GPU Usage

```bash
watch -n 1 nvidia-smi
```

### Training on CPU (Not Recommended)

```bash
bash configs/cpu_training.sh
```

**Expect**:
- Very slow training (hours → days)
- High RAM usage
- Only suitable for testing

---

## Inference and Testing

### Quick Test

```bash
python utils/inference.py \
    --model outputs/sft_qlora_8b \
    --prompt "Explain machine learning in simple terms." \
    --load-in-4bit
```

### Interactive Mode

```bash
python utils/inference.py \
    --model outputs/sft_qlora_8b \
    --interactive \
    --load-in-4bit
```

### Python API

```python
from utils.inference import InferenceEngine

# Initialize
engine = InferenceEngine(
    model_path="outputs/sft_qlora_8b",
    load_in_4bit=True
)

# Generate response
response = engine.chat(
    instruction="Write a Python function to sort a list",
    max_new_tokens=512,
    temperature=0.7
)

print(response)

# Compare multiple prompts
prompts = [
    "What is AI?",
    "Explain neural networks.",
    "Write hello world in Python."
]
engine.compare_responses(prompts)
```

### Testing with LoRA Adapter

```python
engine = InferenceEngine(
    model_path="outputs/sft_qlora_8b",
    base_model_path="apertus/apertus-8b",
    load_in_4bit=True
)
```

---

## Advanced Usage

### Custom Training Script

```python
from transformers import AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# Load model with your custom settings
model = AutoModelForCausalLM.from_pretrained(...)

# Configure LoRA
peft_config = LoraConfig(
    r=32,  # Higher rank
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

# Your custom training logic
trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=peft_config,
    # ... other configs
)

trainer.train()
```

### Multi-Stage Training Pipeline

```bash
# Stage 1: SFT
bash configs/sft_qlora_single_gpu.sh

# Stage 2: DPO on SFT model
bash configs/dpo_qlora.sh

# Stage 3: Test
python utils/inference.py --model outputs/dpo_qlora_8b --interactive
```

### Merging LoRA Adapters

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base + adapter
base_model = AutoModelForCausalLM.from_pretrained("apertus/apertus-8b")
model = PeftModel.from_pretrained(base_model, "outputs/sft_qlora_8b")

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("outputs/sft_merged")
```

### Hyperparameter Tuning

Key hyperparameters to experiment with:

**LoRA**:
- `lora_r`: 8, 16, 32, 64 (higher = more capacity, more memory)
- `lora_alpha`: Usually 2x of `lora_r`
- `lora_dropout`: 0.05, 0.1 (regularization)

**Training**:
- `learning_rate`: 1e-5 to 5e-4 (lower for DPO)
- `num_epochs`: 1-5 (more epochs risk overfitting)
- `batch_size`: Balance memory vs speed

**DPO/ORPO**:
- `beta` (DPO) or `lambda_orpo`: 0.01 to 0.5 (controls preference strength)

---

## Troubleshooting

### Out of Memory (OOM)

**Error**: `CUDA out of memory`

**Solutions**:
1. Enable 4-bit quantization: `--use_4bit true`
2. Reduce batch size: `--per_device_train_batch_size 1`
3. Increase gradient accumulation: `--gradient_accumulation_steps 16`
4. Reduce max length: `--max_seq_length 1024`
5. Use smaller LoRA rank: `--lora_r 8`

### Slow Training

**Issue**: Training is taking too long

**Solutions**:
1. Increase batch size if memory allows
2. Reduce gradient accumulation steps
3. Use smaller validation split
4. Disable W&B logging
5. Use fewer logging/save steps

### Poor Model Quality

**Issue**: Model gives bad responses

**Solutions**:
1. **More data**: Collect 10K+ high-quality examples
2. **Better data**: Review and improve dataset quality
3. **More epochs**: Try 3-5 epochs instead of 1
4. **Lower learning rate**: Try 1e-5 instead of 2e-4
5. **Check format**: Ensure dataset matches expected format
6. **Start from SFT**: Don't skip SFT before DPO

### Installation Issues

**flash-attn won't install**:
```bash
# Comment it out in requirements.txt or skip it
pip install -r requirements.txt || true
```

**bitsandbytes issues**:
```bash
# Ensure CUDA is properly installed
pip install bitsandbytes --force-reinstall
```

### Model Not Loading

**Error**: `Model not found`

**Solutions**:
1. Check model name: `huggingface-cli download apertus/apertus-8b`
2. Login to HF: `huggingface-cli login`
3. Check internet connection
4. Use absolute paths

---

## Best Practices

### Data Quality

- **Quality > Quantity**: 1K high-quality examples > 10K low-quality
- **Diverse examples**: Cover various topics, styles, lengths
- **Proofread**: Review for errors, inconsistencies
- **Balance**: Mix easy and hard examples

### Training Strategy

1. **Start small**: Test with 100 examples first
2. **Monitor closely**: Watch loss curves, check outputs
3. **Save checkpoints**: Don't lose progress
4. **Validate**: Always use validation split
5. **Compare**: Test against base model

### Recommended Pipeline

**For best results:**

```
1. Collect/create 10K+ SFT examples
2. Train SFT model (3 epochs)
3. Test and evaluate SFT model
4. Collect 5K+ preference pairs
5. Train DPO on SFT model (1 epoch)
6. Test and compare to SFT
7. Deploy best model
```

**Faster alternative:**

```
1. Collect 10K+ preference pairs (with chosen/rejected)
2. Train ORPO directly (2 epochs)
3. Test and deploy
```

---

## Resources

### Documentation

- [Hugging Face TRL](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Apertus Model](https://huggingface.co/apertus)

### Papers

- **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **QLoRA**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **DPO**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- **ORPO**: [ORPO: Odds Ratio Preference Optimization](https://arxiv.org/abs/2403.07691)

### Community

- [Hugging Face Forums](https://discuss.huggingface.co/)
- [Apertus Discord/Community](#) (check model card)

---

## FAQ

**Q: Can I train on CPU?**
A: Technically yes, but it's 50-100x slower. Only practical for tiny datasets or testing.

**Q: How much data do I need?**
A: Minimum 1K examples, recommended 10K+ for good results.

**Q: What's the difference between LoRA and QLoRA?**
A: QLoRA adds 4-bit quantization to LoRA, using less memory (10-12GB vs 20-24GB).

**Q: Should I use SFT then DPO, or just ORPO?**
A: ORPO is faster (one stage), SFT→DPO gives slightly better control. Try both!

**Q: How long does training take?**
A: On single GPU with 10K examples: SFT ~2-4 hours, DPO ~1-2 hours, ORPO ~2-3 hours.

**Q: Can I fine-tune for specific domains?**
A: Absolutely! Just create a domain-specific dataset.

**Q: How do I deploy the model?**
A: Use transformers' `pipeline`, vLLM, or hosting services like HF Inference API.

---

## License

This framework is provided as-is for educational and research purposes. Check the Apertus model license for model-specific terms.

---

## Contributing

Found a bug or want to improve this framework? Contributions welcome!

---

## Acknowledgments

- Apertus team for the base model
- Hugging Face for transformers, PEFT, and TRL
- The open-source LLM community

---

**Happy Training! 🚀**

For questions or issues, please review the troubleshooting section or check the documentation links above.
