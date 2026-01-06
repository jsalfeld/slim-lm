#!/bin/bash
# CPU training configuration (VERY SLOW - only for testing/small datasets)
# Recommended: Use cloud GPU instead (vast.ai, RunPod, etc.)

# WARNING: This will be extremely slow. Only use for:
# - Testing your setup
# - Very small datasets (< 100 examples)
# - Understanding the training process

python scripts/train_sft.py \
    --model_name "apertus/apertus-8b" \
    --dataset_path "datasets/sft_template.jsonl" \
    --output_dir "outputs/sft_cpu_test" \
    --use_lora true \
    --lora_r 8 \
    --lora_alpha 16 \
    --use_4bit false \
    --use_8bit false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --max_seq_length 512 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --save_steps 50 \
    --validation_split 0 \
    --use_wandb false \
    --seed 42

# Note: Even with these minimal settings, expect:
# - 10-100x slower than GPU training
# - High RAM usage (32GB+ recommended)
# - Many hours for even small datasets
