#!/bin/bash
# SFT training with LoRA (no quantization) on large GPU (40-80GB VRAM)

python scripts/train_sft.py \
    --model_name "apertus/apertus-8b" \
    --dataset_path "datasets/sft_template.jsonl" \
    --output_dir "outputs/sft_lora_8b" \
    --use_lora true \
    --lora_r 32 \
    --lora_alpha 64 \
    --use_4bit false \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --max_seq_length 4096 \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --validation_split 0.1 \
    --use_flash_attention true \
    --use_wandb false \
    --seed 42
