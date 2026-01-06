"""
Data preprocessing utilities for post-training datasets
"""

import json
import jsonlines
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write_all(data)


def split_dataset(
    data: List[Dict[str, Any]],
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split dataset into train and validation sets."""
    if len(data) < 10:
        print("Warning: Dataset too small for splitting. Using all data for training.")
        return data, []

    train, val = train_test_split(data, test_size=test_size, random_state=random_state)
    return train, val


def convert_csv_to_sft(
    csv_path: str,
    output_path: str,
    instruction_col: str = "instruction",
    input_col: str = "input",
    output_col: str = "output"
):
    """Convert CSV to SFT JSONL format."""
    df = pd.read_csv(csv_path)

    data = []
    for _, row in df.iterrows():
        example = {
            "instruction": str(row[instruction_col]),
            "input": str(row.get(input_col, "")) if input_col in df.columns else "",
            "output": str(row[output_col])
        }
        data.append(example)

    save_jsonl(data, output_path)
    print(f"Converted {len(data)} examples from {csv_path} to {output_path}")


def convert_csv_to_dpo(
    csv_path: str,
    output_path: str,
    prompt_col: str = "prompt",
    chosen_col: str = "chosen",
    rejected_col: str = "rejected"
):
    """Convert CSV to DPO JSONL format."""
    df = pd.read_csv(csv_path)

    data = []
    for _, row in df.iterrows():
        example = {
            "prompt": str(row[prompt_col]),
            "chosen": str(row[chosen_col]),
            "rejected": str(row[rejected_col])
        }
        data.append(example)

    save_jsonl(data, output_path)
    print(f"Converted {len(data)} preference pairs from {csv_path} to {output_path}")


def convert_csv_to_kto(
    csv_path: str,
    output_path: str,
    prompt_col: str = "prompt",
    completion_col: str = "completion",
    label_col: str = "label"
):
    """Convert CSV to KTO JSONL format."""
    df = pd.read_csv(csv_path)

    data = []
    for _, row in df.iterrows():
        label_value = row[label_col]
        # Convert various representations to boolean
        if isinstance(label_value, str):
            label = label_value.lower() in ['true', 'yes', '1', 'good']
        else:
            label = bool(label_value)

        example = {
            "prompt": str(row[prompt_col]),
            "completion": str(row[completion_col]),
            "label": label
        }
        data.append(example)

    save_jsonl(data, output_path)
    print(f"Converted {len(data)} labeled examples from {csv_path} to {output_path}")


def validate_sft_dataset(file_path: str) -> bool:
    """Validate SFT dataset format."""
    data = load_jsonl(file_path)

    required_fields = ["instruction", "output"]
    for i, example in enumerate(data):
        for field in required_fields:
            if field not in example:
                print(f"Error at index {i}: Missing required field '{field}'")
                return False

        if "input" not in example:
            print(f"Warning at index {i}: Missing 'input' field (optional)")

    print(f"✓ SFT dataset valid: {len(data)} examples")
    return True


def validate_dpo_dataset(file_path: str) -> bool:
    """Validate DPO dataset format."""
    data = load_jsonl(file_path)

    required_fields = ["prompt", "chosen", "rejected"]
    for i, example in enumerate(data):
        for field in required_fields:
            if field not in example:
                print(f"Error at index {i}: Missing required field '{field}'")
                return False

    print(f"✓ DPO dataset valid: {len(data)} preference pairs")
    return True


def validate_kto_dataset(file_path: str) -> bool:
    """Validate KTO dataset format."""
    data = load_jsonl(file_path)

    required_fields = ["prompt", "completion", "label"]
    for i, example in enumerate(data):
        for field in required_fields:
            if field not in example:
                print(f"Error at index {i}: Missing required field '{field}'")
                return False

        if not isinstance(example["label"], bool):
            print(f"Error at index {i}: 'label' must be boolean (true/false)")
            return False

    print(f"✓ KTO dataset valid: {len(data)} labeled examples")
    return True


def merge_datasets(file_paths: List[str], output_path: str):
    """Merge multiple JSONL datasets into one."""
    all_data = []
    for file_path in file_paths:
        data = load_jsonl(file_path)
        all_data.extend(data)
        print(f"Loaded {len(data)} examples from {file_path}")

    save_jsonl(all_data, output_path)
    print(f"Merged {len(all_data)} total examples to {output_path}")


def deduplicate_dataset(file_path: str, output_path: str, key: str = "instruction"):
    """Remove duplicate entries based on a key field."""
    data = load_jsonl(file_path)

    seen = set()
    unique_data = []

    for example in data:
        key_value = example.get(key, "")
        if key_value not in seen:
            seen.add(key_value)
            unique_data.append(example)

    save_jsonl(unique_data, output_path)
    print(f"Removed {len(data) - len(unique_data)} duplicates")
    print(f"Saved {len(unique_data)} unique examples to {output_path}")


def sample_dataset(
    file_path: str,
    output_path: str,
    n_samples: int = 1000,
    random_state: int = 42
):
    """Sample a subset of the dataset."""
    data = load_jsonl(file_path)

    if len(data) <= n_samples:
        print(f"Dataset has {len(data)} examples, less than requested {n_samples}")
        save_jsonl(data, output_path)
    else:
        import random
        random.seed(random_state)
        sampled = random.sample(data, n_samples)
        save_jsonl(sampled, output_path)
        print(f"Sampled {n_samples} examples from {len(data)} total")


def analyze_dataset(file_path: str):
    """Analyze dataset statistics."""
    data = load_jsonl(file_path)

    print(f"\n{'='*50}")
    print(f"Dataset Analysis: {file_path}")
    print(f"{'='*50}")
    print(f"Total examples: {len(data)}")

    if not data:
        return

    # Get field names
    fields = list(data[0].keys())
    print(f"Fields: {', '.join(fields)}")

    # Analyze text lengths
    for field in fields:
        if isinstance(data[0].get(field), str):
            lengths = [len(str(ex.get(field, ""))) for ex in data]
            avg_length = sum(lengths) / len(lengths)
            max_length = max(lengths)
            min_length = min(lengths)

            print(f"\n{field}:")
            print(f"  Average length: {avg_length:.1f} chars")
            print(f"  Min length: {min_length} chars")
            print(f"  Max length: {max_length} chars")

    print(f"{'='*50}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data preprocessing utilities")
    parser.add_argument("command", choices=[
        "validate-sft", "validate-dpo", "validate-kto",
        "split", "merge", "deduplicate", "sample", "analyze",
        "csv-to-sft", "csv-to-dpo", "csv-to-kto"
    ])
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--n-samples", type=int, default=1000)

    args = parser.parse_args()

    if args.command == "validate-sft":
        validate_sft_dataset(args.input)
    elif args.command == "validate-dpo":
        validate_dpo_dataset(args.input)
    elif args.command == "validate-kto":
        validate_kto_dataset(args.input)
    elif args.command == "split":
        data = load_jsonl(args.input)
        train, val = split_dataset(data, args.test_size)
        save_jsonl(train, args.output or "train.jsonl")
        save_jsonl(val, args.output.replace("train", "val") if args.output else "val.jsonl")
    elif args.command == "analyze":
        analyze_dataset(args.input)
    elif args.command == "sample":
        sample_dataset(args.input, args.output, args.n_samples)
    else:
        print(f"Command {args.command} not fully implemented in CLI mode")
        print("Use as a Python module for more functionality")
