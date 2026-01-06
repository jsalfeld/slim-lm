# Post-Training Cheatsheet

Quick reference for common commands and configurations.

## Installation

```bash
./setup.sh                    # Run setup script
source .venv/bin/activate     # Activate environment
huggingface-cli login         # Login to HF
wandb login                   # Login to W&B (optional)
```

## Training Commands

### SFT (Supervised Fine-Tuning)
```bash
# QLoRA on 12GB GPU
bash configs/sft_qlora_single_gpu.sh

# LoRA on 40GB GPU
bash configs/sft_lora_large_gpu.sh

# Custom
python scripts/train_sft.py \
    --model_name "apertus/apertus-8b" \
    --dataset_path "datasets/my_data.jsonl" \
    --output_dir "outputs/my_model" \
    --use_4bit true \
    --num_train_epochs 3
```

### DPO (Direct Preference Optimization)
```bash
# QLoRA
bash configs/dpo_qlora.sh

# Custom
python scripts/train_dpo.py \
    --model_name "outputs/sft_model" \
    --dataset_path "datasets/dpo_data.jsonl" \
    --output_dir "outputs/dpo_model" \
    --beta 0.1
```

### ORPO (Odds Ratio Preference Optimization)
```bash
# QLoRA
bash configs/orpo_qlora.sh

# Custom
python scripts/train_orpo.py \
    --model_name "apertus/apertus-8b" \
    --dataset_path "datasets/dpo_data.jsonl" \
    --output_dir "outputs/orpo_model" \
    --lambda_orpo 0.1
```

## Data Preparation

### Validate Dataset
```bash
python utils/data_prep.py validate-sft --input datasets/my_data.jsonl
python utils/data_prep.py validate-dpo --input datasets/my_dpo.jsonl
```

### Analyze Dataset
```bash
python utils/data_prep.py analyze --input datasets/my_data.jsonl
```

### Split Dataset
```bash
python utils/data_prep.py split --input data.jsonl --output train.jsonl --test-size 0.1
```

### Convert CSV to JSONL
```python
from utils.data_prep import convert_csv_to_sft

convert_csv_to_sft(
    "data.csv",
    "datasets/sft_data.jsonl",
    instruction_col="question",
    output_col="answer"
)
```

## Inference

### Command Line
```bash
# Interactive
python utils/inference.py --model outputs/my_model --interactive --load-in-4bit

# Single prompt
python utils/inference.py --model outputs/my_model --prompt "Your question" --load-in-4bit
```

### Python
```python
from utils.inference import InferenceEngine

engine = InferenceEngine("outputs/my_model", load_in_4bit=True)
response = engine.chat("Your question")
print(response)
```

## Key Parameters

### Memory Management
| Setting | 10GB VRAM | 16GB VRAM | 24GB VRAM | 40GB+ VRAM |
|---------|-----------|-----------|-----------|------------|
| use_4bit | true | true | false | false |
| batch_size | 1-2 | 4 | 8 | 16 |
| max_seq_length | 1024 | 2048 | 4096 | 4096 |
| lora_r | 8 | 16 | 32 | 64 |
| grad_accum | 16 | 8 | 4 | 2 |

### Learning Rates
- **SFT**: 2e-4 (typical)
- **DPO**: 5e-5 (lower than SFT)
- **ORPO**: 8e-6 (lowest)

### Training Epochs
- **SFT**: 3-5 epochs
- **DPO**: 1-2 epochs
- **ORPO**: 2-3 epochs

## Dataset Formats

### SFT Format
```json
{"instruction": "Question", "input": "Context (optional)", "output": "Answer"}
```

### DPO/ORPO Format
```json
{"prompt": "Question", "chosen": "Good answer", "rejected": "Bad answer"}
```

### KTO Format
```json
{"prompt": "Question", "completion": "An answer", "label": true}
```

## Monitoring

### TensorBoard
```bash
tensorboard --logdir outputs/my_model/runs
```

### GPU Monitoring
```bash
watch -n 1 nvidia-smi
```

### Check Training Progress
```bash
tail -f outputs/my_model/trainer_log.txt
```

## Quick Fixes

### Out of Memory
```bash
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--max_seq_length 1024 \
--lora_r 8
```

### Slow Training
```bash
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 2 \
--logging_steps 50 \
--save_steps 500
```

### Poor Quality
- Collect more data (10K+ examples)
- Train longer (5 epochs)
- Lower learning rate (1e-5)
- Review dataset quality

## File Paths

```
datasets/
  ├── sft_template.jsonl      # SFT examples
  ├── dpo_template.jsonl      # DPO/ORPO examples
  └── kto_template.jsonl      # KTO examples

scripts/
  ├── train_sft.py            # SFT training
  ├── train_dpo.py            # DPO training
  └── train_orpo.py           # ORPO training

configs/
  ├── sft_qlora_single_gpu.sh # SFT config
  ├── dpo_qlora.sh            # DPO config
  └── orpo_qlora.sh           # ORPO config

utils/
  ├── data_prep.py            # Data utilities
  └── inference.py            # Inference engine

outputs/
  └── my_model/               # Trained models
```

## Common Workflows

### Workflow 1: SFT → DPO
```bash
# 1. Train SFT
bash configs/sft_qlora_single_gpu.sh

# 2. Test SFT
python utils/inference.py --model outputs/sft_qlora_8b --interactive

# 3. Create DPO dataset with chosen/rejected pairs

# 4. Train DPO
bash configs/dpo_qlora.sh

# 5. Compare
python utils/inference.py --model outputs/dpo_qlora_8b --interactive
```

### Workflow 2: ORPO Only
```bash
# 1. Create preference dataset

# 2. Train ORPO
bash configs/orpo_qlora.sh

# 3. Test
python utils/inference.py --model outputs/orpo_qlora_8b --interactive
```

### Workflow 3: Iterative Improvement
```bash
# 1. Train with small dataset
python scripts/train_sft.py --dataset_path small.jsonl --output_dir v1

# 2. Test and identify weaknesses
python utils/inference.py --model v1 --interactive

# 3. Add more training data for weak areas

# 4. Continue training
python scripts/train_sft.py --model_name v1 --dataset_path improved.jsonl --output_dir v2
```

## Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0           # Use GPU 0
export WANDB_PROJECT="my-project"       # W&B project
export HF_HOME="./cache"                # HF cache location
export TOKENIZERS_PARALLELISM=false     # Disable tokenizer warnings
```

## Useful Commands

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Count examples in dataset
wc -l datasets/my_data.jsonl

# View first example
head -n 1 datasets/my_data.jsonl | python -m json.tool

# Merge datasets
cat data1.jsonl data2.jsonl > combined.jsonl

# Sample dataset
shuf -n 1000 big_data.jsonl > sample.jsonl
```

## Troubleshooting Steps

1. **Check dataset format**: `python utils/data_prep.py validate-sft --input data.jsonl`
2. **Test with small data**: Use 100 examples first
3. **Monitor GPU**: `watch -n 1 nvidia-smi`
4. **Check logs**: `tail -f outputs/model/trainer_log.txt`
5. **Reduce memory**: Lower batch size, sequence length, LoRA rank
6. **Ask for help**: Check README.md troubleshooting section

## Performance Tips

1. **Use QLoRA for memory efficiency** (4-bit quantization)
2. **Batch size × grad_accum = effective batch size** (keep total ~16-32)
3. **Use flash attention** if you have Ampere+ GPU (A100, RTX 30xx/40xx)
4. **Gradient checkpointing** is automatic with QLoRA
5. **Monitor loss curves** - should decrease steadily
6. **Validate regularly** - catch overfitting early

---

**Need more details?** Check:
- `README.md` - Full documentation
- `QUICKSTART.md` - Getting started
- `datasets/README_datasets.md` - Dataset formats
