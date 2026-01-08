#!/usr/bin/env python3
"""
Evaluation script for trained models.
Generates performance plots and statistics.

Usage:
    python evaluate.py --model outputs/sft --data data/eval.jsonl
    python evaluate.py --model outputs/sft --data data/eval.jsonl --compare outputs/dpo
"""

import argparse
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_model(model_path: str, device: str = "auto"):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    return model, tokenizer


def calculate_perplexity(model, tokenizer, texts: List[str], max_length: int = 512) -> List[float]:
    """Calculate perplexity for each text."""
    perplexities = []

    for text in tqdm(texts, desc="Calculating perplexity"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            perplexity = np.exp(loss)
            perplexities.append(perplexity)

    return perplexities


def generate_responses(model, tokenizer, prompts: List[str], max_new_tokens: int = 256) -> List[str]:
    """Generate responses for prompts."""
    responses = []

    for prompt in tqdm(prompts, desc="Generating responses"):
        formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        responses.append(response)

    return responses


def load_training_metrics(model_path: str) -> Optional[dict]:
    """Load training metrics if available."""
    metrics_path = Path(model_path) / "training_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def plot_training_curves(metrics: dict, output_path: str, title: str = "Training Curves"):
    """Plot training loss and eval loss curves."""
    log_history = metrics.get("log_history", [])

    # Extract losses
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    fig, ax = plt.subplots(figsize=(10, 6))

    if train_losses:
        ax.plot(train_steps, train_losses, label="Training Loss", color="blue", alpha=0.7)
    if eval_losses:
        ax.plot(eval_steps, eval_losses, label="Eval Loss", color="red", marker="o", markersize=4)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Training curves saved to {output_path}")


def plot_perplexity_distribution(perplexities: List[float], output_path: str, title: str = "Perplexity Distribution"):
    """Plot perplexity histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(perplexities, bins=30, color="steelblue", edgecolor="white", alpha=0.7)
    ax.axvline(np.mean(perplexities), color="red", linestyle="--", label=f"Mean: {np.mean(perplexities):.2f}")
    ax.axvline(np.median(perplexities), color="green", linestyle="--", label=f"Median: {np.median(perplexities):.2f}")

    ax.set_xlabel("Perplexity")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Perplexity distribution saved to {output_path}")


def plot_model_comparison(results: dict, output_path: str):
    """Plot comparison between multiple models."""
    models = list(results.keys())
    metrics = ["mean_perplexity", "median_perplexity", "std_perplexity"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in models]
        axes[i].bar(models, values, color=plt.cm.tab10.colors[:len(models)])
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].set_title(metric.replace("_", " ").title())
        axes[i].tick_params(axis="x", rotation=45)

        for j, v in enumerate(values):
            axes[i].text(j, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Model comparison saved to {output_path}")


def plot_response_lengths(responses: List[str], output_path: str, title: str = "Response Length Distribution"):
    """Plot response length distribution."""
    lengths = [len(r.split()) for r in responses]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(lengths, bins=30, color="coral", edgecolor="white", alpha=0.7)
    ax.axvline(np.mean(lengths), color="red", linestyle="--", label=f"Mean: {np.mean(lengths):.1f}")

    ax.set_xlabel("Response Length (words)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Response lengths saved to {output_path}")


def calculate_stats(perplexities: List[float]) -> dict:
    """Calculate statistics from perplexities."""
    return {
        "mean_perplexity": float(np.mean(perplexities)),
        "median_perplexity": float(np.median(perplexities)),
        "std_perplexity": float(np.std(perplexities)),
        "min_perplexity": float(np.min(perplexities)),
        "max_perplexity": float(np.max(perplexities)),
        "num_samples": len(perplexities),
    }


def evaluate_model(model_path: str, data_path: str, output_dir: str, max_samples: int = 100):
    """Full evaluation of a single model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model(model_path)

    # Load data
    dataset = load_dataset("json", data_files=data_path, split="train")
    if len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    # Get texts for perplexity
    if "output" in dataset.column_names:
        texts = [f"{ex['instruction']} {ex['output']}" for ex in dataset]
    elif "chosen" in dataset.column_names:
        texts = [f"{ex['prompt']} {ex['chosen']}" for ex in dataset]
    else:
        texts = [ex.get("text", str(ex)) for ex in dataset]

    # Calculate perplexity
    perplexities = calculate_perplexity(model, tokenizer, texts)
    stats = calculate_stats(perplexities)

    # Generate responses
    prompts = [ex.get("instruction", ex.get("prompt", "")) for ex in dataset[:min(20, len(dataset))]]
    responses = generate_responses(model, tokenizer, prompts)

    # Plot training curves if available
    metrics = load_training_metrics(model_path)
    if metrics:
        plot_training_curves(metrics, str(output_dir / "training_curves.png"))

    # Plot perplexity distribution
    plot_perplexity_distribution(perplexities, str(output_dir / "perplexity_distribution.png"))

    # Plot response lengths
    plot_response_lengths(responses, str(output_dir / "response_lengths.png"))

    # Save stats
    stats_path = output_dir / "evaluation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Save sample responses
    samples_path = output_dir / "sample_responses.json"
    samples = [{"prompt": p, "response": r} for p, r in zip(prompts, responses)]
    with open(samples_path, "w") as f:
        json.dump(samples, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Samples evaluated: {stats['num_samples']}")
    print(f"Mean Perplexity: {stats['mean_perplexity']:.2f}")
    print(f"Median Perplexity: {stats['median_perplexity']:.2f}")
    print(f"Std Perplexity: {stats['std_perplexity']:.2f}")
    print(f"Min/Max Perplexity: {stats['min_perplexity']:.2f} / {stats['max_perplexity']:.2f}")
    print("=" * 50)
    print(f"\nResults saved to {output_dir}")

    return stats


def compare_models(model_paths: List[str], data_path: str, output_dir: str, max_samples: int = 100):
    """Compare multiple models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for model_path in model_paths:
        model_name = Path(model_path).name
        print(f"\nEvaluating {model_name}...")

        model_output_dir = output_dir / model_name
        stats = evaluate_model(model_path, data_path, str(model_output_dir), max_samples)
        results[model_name] = stats

    # Plot comparison
    plot_model_comparison(results, str(output_dir / "model_comparison.png"))

    # Save comparison results
    comparison_path = output_dir / "comparison_results.json"
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    df = pd.DataFrame(results).T
    print(df.to_string())
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--data", required=True, help="Path to evaluation data (JSONL)")
    parser.add_argument("--output", default="eval_results", help="Output directory for results")
    parser.add_argument("--compare", nargs="+", help="Additional models to compare")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples to evaluate")
    args = parser.parse_args()

    if args.compare:
        # Compare multiple models
        all_models = [args.model] + args.compare
        compare_models(all_models, args.data, args.output, args.max_samples)
    else:
        # Single model evaluation
        evaluate_model(args.model, args.data, args.output, args.max_samples)


if __name__ == "__main__":
    main()
