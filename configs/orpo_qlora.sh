#!/bin/bash
# ORPO training with QLoRA (single-stage SFT + alignment)

python scripts/train_orpo.py \
    --model_name "apertus/apertus-8b" \
    --dataset_path "datasets/dpo_template.jsonl" \
    --output_dir "outputs/orpo_qlora_8b" \
    --use_lora true \
    --lora_r 16 \
    --lora_alpha 32 \
    --use_4bit true \
    --lambda_orpo 0.1 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 8e-6 \
    --max_seq_length 2048 \
    --max_prompt_length 1024 \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --validation_split 0.1 \
    --use_wandb false \
    --seed 42
