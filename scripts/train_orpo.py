"""
ORPO (Odds Ratio Preference Optimization) Training Script
Combines SFT and preference learning in a single stage
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    HfArgumentParser,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import ORPOTrainer, ORPOConfig
from datasets import load_dataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class ScriptArguments:
    """Arguments for ORPO training."""

    # Model arguments
    model_name: str = field(
        default="apertus/apertus-8b",  # Can start from base model (doesn't need SFT first)
        metadata={"help": "The model checkpoint to use"}
    )

    # Data arguments
    dataset_path: str = field(
        default="datasets/dpo_template.jsonl",  # Uses same format as DPO
        metadata={"help": "Path to preference dataset"}
    )
    validation_split: float = field(
        default=0.1,
        metadata={"help": "Validation split fraction"}
    )

    # LoRA arguments
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Target modules for LoRA"}
    )

    # Quantization
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Use 4-bit quantization"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype"}
    )

    # ORPO-specific arguments
    lambda_orpo: float = field(
        default=0.1,
        metadata={"help": "ORPO lambda parameter (weight for odds ratio loss)"}
    )

    # Training arguments
    output_dir: str = field(
        default="outputs/orpo_model",
        metadata={"help": "Output directory"}
    )
    num_train_epochs: int = field(
        default=2,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "Training batch size"}
    )
    per_device_eval_batch_size: int = field(
        default=2,
        metadata={"help": "Eval batch size"}
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=8e-6,
        metadata={"help": "Learning rate"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "Maximum prompt length"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Warmup ratio"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Logging steps"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save steps"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Eval steps"}
    )

    # Other
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Use W&B"}
    )
    wandb_project: str = field(
        default="apertus-orpo",
        metadata={"help": "W&B project"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )


def create_bnb_config(args: ScriptArguments) -> Optional[BitsAndBytesConfig]:
    """Create quantization config."""
    if not args.use_4bit:
        return None

    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def create_peft_config(args: ScriptArguments) -> Optional[LoraConfig]:
    """Create LoRA config."""
    if not args.use_lora:
        return None

    target_modules = args.lora_target_modules.split(",")
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Install with: pip install wandb")
            args.use_wandb = False
        else:
            wandb.init(project=args.wandb_project, config=vars(args))

    print(f"Loading model: {args.model_name}")
    print(f"ORPO lambda: {args.lambda_orpo}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantization config
    bnb_config = create_bnb_config(args)

    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    # Add LoRA
    if args.use_lora:
        peft_config = create_peft_config(args)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # Verify format
    required_columns = ["prompt", "chosen", "rejected"]
    for col in required_columns:
        if col not in dataset.column_names:
            raise ValueError(f"Dataset must contain '{col}' column")

    # Split
    if args.validation_split > 0:
        dataset = dataset.train_test_split(test_size=args.validation_split, seed=args.seed)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    print(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Validation samples: {len(eval_dataset)}")

    # ORPO Config (extends TrainingArguments)
    orpo_config = ORPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        bf16=True,
        tf32=True,
        optim="paged_adamw_8bit" if bnb_config else "adamw_torch",
        report_to="wandb" if args.use_wandb else "none",
        seed=args.seed,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        # ORPO-specific
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.lambda_orpo,  # In ORPOConfig, beta is used for lambda
    )

    # Create ORPO trainer
    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting ORPO training...")
    trainer.train()

    # Save
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluate
    if eval_dataset:
        metrics = trainer.evaluate()
        print("Final metrics:", metrics)

    print("ORPO training complete!")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
