# Quick Start Guide

Get started with post-training in 5 minutes!

## Step 1: Install Dependencies (2 min)

```bash
# Activate your virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Login to Hugging Face (you'll need an account and token)
huggingface-cli login
```

## Step 2: Prepare Your Data (1 min)

### Option A: Use Templates (Fastest)

The project includes template datasets you can use immediately:
- `datasets/sft_template.jsonl` - For SFT training
- `datasets/dpo_template.jsonl` - For DPO/ORPO training

### Option B: Create Your Own

```bash
# Copy template
cp datasets/sft_template.jsonl datasets/my_data.jsonl

# Edit with your examples (JSON Lines format)
# Each line: {"instruction": "...", "input": "...", "output": "..."}
```

## Step 3: Choose Your Training Method (30 seconds)

### For Instruction Following → Use SFT

```bash
bash configs/sft_qlora_single_gpu.sh
```

### For Alignment (Better Responses) → Use DPO

First train SFT, then:
```bash
bash configs/dpo_qlora.sh
```

### For Both in One Step → Use ORPO

```bash
bash configs/orpo_qlora.sh
```

## Step 4: Test Your Model (1 min)

```bash
# Interactive testing
python utils/inference.py \
    --model outputs/sft_qlora_8b \
    --interactive \
    --load-in-4bit
```

Or use Python:

```python
from utils.inference import InferenceEngine

engine = InferenceEngine("outputs/sft_qlora_8b", load_in_4bit=True)
response = engine.chat("Explain quantum computing simply.")
print(response)
```

## Common Scenarios

### Scenario 1: "I want to teach my model new knowledge"

Use **SFT** with instruction-response pairs:

```jsonl
{"instruction": "What is photosynthesis?", "input": "", "output": "Photosynthesis is..."}
{"instruction": "Explain quantum entanglement", "input": "", "output": "Quantum entanglement is..."}
```

```bash
bash configs/sft_qlora_single_gpu.sh
```

### Scenario 2: "I want better quality responses"

Use **DPO** with chosen/rejected pairs:

```jsonl
{"prompt": "How do I learn coding?", "chosen": "Start with...", "rejected": "Just copy code..."}
```

```bash
# First do SFT (if not done)
bash configs/sft_qlora_single_gpu.sh
# Then DPO
bash configs/dpo_qlora.sh
```

### Scenario 3: "I want to do it all in one go"

Use **ORPO**:

```bash
bash configs/orpo_qlora.sh
```

## Hardware-Specific Guides

### I have a 12-16GB GPU (RTX 3090, 4080, T4)

Perfect! Use the default configs with QLoRA:

```bash
bash configs/sft_qlora_single_gpu.sh
```

### I have a 24GB+ GPU (RTX 4090, A10G)

You can use larger batches and sequences:

Edit the config to increase:
- `--per_device_train_batch_size 8`
- `--max_seq_length 4096`

### I only have CPU

Training will be very slow. Only use for small tests:

```bash
bash configs/cpu_training.sh
```

Better option: Use cloud GPU (Vast.ai, RunPod) - costs $0.20-0.50/hour

## Troubleshooting Quick Fixes

### Out of Memory?

```bash
# Reduce batch size and max length
python scripts/train_sft.py \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 1024 \
    [... other args ...]
```

### Training too slow?

```bash
# Increase batch size (if memory allows)
# Reduce logging frequency
--logging_steps 50 \
--save_steps 500
```

### Bad outputs?

1. Check dataset quality
2. Train for more epochs
3. Lower learning rate
4. Collect more data

## Next Steps

1. **Read the full README.md** for detailed explanations
2. **Check datasets/README_datasets.md** for data formatting
3. **Experiment with hyperparameters** in configs/
4. **Monitor training** with W&B or TensorBoard
5. **Evaluate results** using utils/inference.py

## Need Help?

1. Check **README.md** - Comprehensive guide
2. Check **Troubleshooting** section in README
3. Review Hugging Face TRL documentation
4. Check Apertus model card

## Estimated Times (Single GPU)

| Task | Dataset Size | Time |
|------|--------------|------|
| SFT Training | 1K examples | 15-30 min |
| SFT Training | 10K examples | 2-4 hours |
| DPO Training | 5K pairs | 1-2 hours |
| ORPO Training | 10K pairs | 2-3 hours |
| Inference | 1 prompt | 2-5 seconds |

---

**Ready to train? Start with:**

```bash
bash configs/sft_qlora_single_gpu.sh
```

Good luck! 🚀
