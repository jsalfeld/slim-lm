"""
Supervised Fine-Tuning (SFT) Script with LoRA/QLoRA
Supports training Apertus 8B and other models with memory-efficient techniques
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
from trl import SFTTrainer
from datasets import load_dataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class ScriptArguments:
    """Arguments for the training script."""

    # Model arguments
    model_name: str = field(
        default="apertus/apertus-8b",
        metadata={"help": "The model checkpoint to use"}
    )

    # Data arguments
    dataset_path: str = field(
        default="datasets/sft_template.jsonl",
        metadata={"help": "Path to training dataset (JSONL format)"}
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
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )

    # Quantization arguments
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Use 4-bit quantization (QLoRA)"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit quantization (ignored if use_4bit is True)"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4-bit: float16, bfloat16"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type: fp4 or nf4"}
    )

    # Training arguments
    output_dir: str = field(
        default="outputs/sft_model",
        metadata={"help": "Output directory for model and checkpoints"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device during training"}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device during evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "Learning rate"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Warmup ratio"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X steps"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint every X steps"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Evaluate every X steps"}
    )

    # Other arguments
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Use Flash Attention 2 (requires flash-attn package)"}
    )
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Use Weights & Biases for logging (requires wandb package)"}
    )
    wandb_project: str = field(
        default="apertus-sft",
        metadata={"help": "W&B project name"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )


def format_instruction(example):
    """Format the instruction dataset into a prompt."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    return {"text": prompt}


def create_bnb_config(args: ScriptArguments) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytes configuration for quantization."""
    if not (args.use_4bit or args.use_8bit):
        return None

    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    if args.use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif args.use_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)

    return None


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
    print(f"Output directory: {args.output_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Add padding token if not present
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

    if args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )

    # Prepare model for k-bit training
    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    if args.use_lora:
        peft_config = create_peft_config(args)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # Format dataset
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

    # Split into train/validation
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
        bf16=True,  # Use bfloat16 if available
        tf32=True,  # Use TF32 on Ampere GPUs
        optim="paged_adamw_8bit" if bnb_config else "adamw_torch",
        report_to="wandb" if args.use_wandb else "none",
        seed=args.seed,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save metrics
    if eval_dataset:
        metrics = trainer.evaluate()
        print("Final metrics:", metrics)

    print("Training complete!")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
