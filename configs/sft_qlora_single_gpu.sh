#!/bin/bash
# SFT training with QLoRA on a single GPU (12-16GB VRAM)

python scripts/train_sft.py \
    --model_name "apertus/apertus-8b" \
    --dataset_path "datasets/sft_template.jsonl" \
    --output_dir "outputs/sft_qlora_8b" \
    --use_lora true \
    --lora_r 16 \
    --lora_alpha 32 \
    --use_4bit true \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_seq_length 2048 \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --validation_split 0.1 \
    --use_wandb false \
    --seed 42
