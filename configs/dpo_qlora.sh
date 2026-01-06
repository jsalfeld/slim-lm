#!/bin/bash
# DPO training with QLoRA on single GPU

python scripts/train_dpo.py \
    --model_name "outputs/sft_qlora_8b" \
    --dataset_path "datasets/dpo_template.jsonl" \
    --output_dir "outputs/dpo_qlora_8b" \
    --use_lora true \
    --lora_r 16 \
    --lora_alpha 32 \
    --use_4bit true \
    --beta 0.1 \
    --loss_type "sigmoid" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --max_seq_length 2048 \
    --max_prompt_length 1024 \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --validation_split 0.1 \
    --use_wandb false \
    --seed 42
