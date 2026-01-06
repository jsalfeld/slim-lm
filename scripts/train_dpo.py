"""
Direct Preference Optimization (DPO) Training Script
Aligns model using preference pairs without reinforcement learning
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
from trl import DPOTrainer
from datasets import load_dataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class ScriptArguments:
    """Arguments for DPO training."""

    # Model arguments
    model_name: str = field(
        default="outputs/sft_model",  # Usually start from SFT model
        metadata={"help": "The model checkpoint to use (typically an SFT model)"}
    )

    # Data arguments
    dataset_path: str = field(
        default="datasets/dpo_template.jsonl",
        metadata={"help": "Path to DPO dataset (JSONL with prompt, chosen, rejected)"}
    )
    validation_split: float = field(
        default=0.1,
        metadata={"help": "Fraction of data to use for validation"}
    )

    # LoRA arguments
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules"}
    )

    # Quantization arguments
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Use 4-bit quantization"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype: float16, bfloat16"}
    )

    # DPO-specific arguments
    beta: float = field(
        default=0.1,
        metadata={"help": "DPO beta parameter (temperature for preference distribution)"}
    )
    loss_type: str = field(
        default="sigmoid",
        metadata={"help": "Loss type: sigmoid, hinge, ipo, or kto_pair"}
    )

    # Training arguments
    output_dir: str = field(
        default="outputs/dpo_model",
        metadata={"help": "Output directory"}
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs (DPO typically needs fewer)"}
    )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size per device (DPO uses more memory)"}
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
        default=5e-5,
        metadata={"help": "Learning rate (lower than SFT)"}
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
        metadata={"help": "Logging frequency"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint frequency"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Evaluation frequency"}
    )

    # Other arguments
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Use W&B for logging"}
    )
    wandb_project: str = field(
        default="apertus-dpo",
        metadata={"help": "W&B project name"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )


def create_bnb_config(args: ScriptArguments) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytes configuration."""
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
    """Create PEFT (LoRA) configuration."""
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
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Initialize W&B
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Install with: pip install wandb")
            args.use_wandb = False
        else:
            wandb.init(project=args.wandb_project, config=vars(args))

    print(f"Loading model: {args.model_name}")
    print(f"Using 4-bit: {args.use_4bit}, Using LoRA: {args.use_lora}")
    print(f"DPO Beta: {args.beta}, Loss type: {args.loss_type}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create quantization config
    bnb_config = create_bnb_config(args)

    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # Prepare for k-bit training
    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    # Add LoRA
    if args.use_lora:
        peft_config = create_peft_config(args)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Load reference model (frozen copy for DPO)
    # In DPO, we need both the trainable model and a reference model
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )

    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # Verify dataset format
    required_columns = ["prompt", "chosen", "rejected"]
    for col in required_columns:
        if col not in dataset.column_names:
            raise ValueError(f"Dataset must contain '{col}' column. Found: {dataset.column_names}")

    # Split dataset
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

    # Training arguments
    training_args = TrainingArguments(
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
        remove_unused_columns=False,  # Important for DPO
    )

    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
    )

    # Train
    print("Starting DPO training...")
    trainer.train()

    # Save final model
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluate
    if eval_dataset:
        metrics = trainer.evaluate()
        print("Final metrics:", metrics)

    print("DPO training complete!")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
