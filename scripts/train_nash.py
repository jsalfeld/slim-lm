#!/usr/bin/env python3
"""Nash-MD (Nash Learning from Human Feedback) training script."""

import argparse
import yaml
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import NashMDConfig, NashMDTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Nash-MD Training | Model: {cfg['model_name']}")

    # Quantization
    quant_config = None
    if cfg.get("use_qlora"):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA
    if quant_config:
        model = prepare_model_for_kbit_training(model)

    if cfg.get("use_lora"):
        lora_config = LoraConfig(
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Data - Nash only needs prompts
    train_data = load_dataset("json", data_files=cfg["train_data"], split="train")
    eval_data = load_dataset("json", data_files=cfg["eval_data"], split="train") if cfg.get("eval_data") else None

    def extract_prompt(ex):
        return {"prompt": ex.get("prompt", ex.get("instruction", ""))}

    train_data = train_data.map(extract_prompt)
    if eval_data:
        eval_data = eval_data.map(extract_prompt)

    # Judge function (replace with reward model for production)
    def judge_fn(completions, prompts=None, **kwargs):
        """Placeholder judge. Replace with actual reward model."""
        return [min(len(c) / 100, 2.0) for c in completions]

    # Train
    training_config = NashMDConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("num_epochs", 1),
        per_device_train_batch_size=cfg.get("batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 5e-7),
        max_new_tokens=cfg.get("max_new_tokens", 256),
        max_length=cfg.get("max_length", 512),
        temperature=cfg.get("temperature", 0.9),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        bf16=True,
        beta=cfg.get("beta", [0.1]),
        mixture_coef=cfg.get("mixture_coef", [0.5, 0.5]),
        loss_type=cfg.get("loss_type", "sigmoid"),
        optim="paged_adamw_8bit" if cfg.get("use_qlora") else "adamw_torch",
        report_to="none",
    )

    trainer = NashMDTrainer(
        model=model,
        args=training_config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        judge=judge_fn,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Model saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
