# Post-Training Dataset Formats

This directory contains template datasets for different post-training methods.

## Dataset Formats

### 1. SFT (Supervised Fine-Tuning)
**File**: `sft_template.jsonl`

**Format**:
```json
{
  "instruction": "The task description or question",
  "input": "Optional context or input (can be empty string)",
  "output": "The desired response"
}
```

**Use Case**: Training the model to follow instructions and respond appropriately.

**Data Collection Tips**:
- Use diverse instruction types (questions, commands, requests)
- Include various domains (code, math, reasoning, creative writing)
- Ensure high-quality, accurate outputs
- Typical dataset size: 10K-100K examples

---

### 2. DPO (Direct Preference Optimization)
**File**: `dpo_template.jsonl`

**Format**:
```json
{
  "prompt": "The input prompt/question",
  "chosen": "The preferred/better response",
  "rejected": "The less preferred/worse response"
}
```

**Use Case**: Aligning the model to prefer certain response styles or qualities.

**Data Collection Tips**:
- Chosen responses should be helpful, harmless, and honest
- Rejected responses can be: incorrect, harmful, irrelevant, or poorly formatted
- Clear quality difference between chosen/rejected
- Typical dataset size: 5K-50K pairs

---

### 3. KTO (Kahneman-Tversky Optimization)
**File**: `kto_template.jsonl`

**Format**:
```json
{
  "prompt": "The input prompt",
  "completion": "A response to evaluate",
  "label": true/false
}
```

**Use Case**: Binary feedback on whether a completion is good or bad (simpler than DPO).

**Data Collection Tips**:
- Label true for good responses, false for bad ones
- Don't need paired comparisons like DPO
- Can use thumbs up/down feedback
- Typical dataset size: 10K-100K examples

---

## Creating Your Own Datasets

### Method 1: Manual Curation
1. Write examples manually based on your use case
2. Use the templates as a starting point
3. Ensure diversity and quality

### Method 2: Using Existing Data
```python
# Convert from common formats
import jsonlines

# From CSV
import pandas as pd
df = pd.read_csv('your_data.csv')
with jsonlines.open('sft_data.jsonl', 'w') as writer:
    for _, row in df.iterrows():
        writer.write({
            'instruction': row['question'],
            'input': '',
            'output': row['answer']
        })
```

### Method 3: Using LLM to Generate
```python
# Use a strong model to generate training data
# Example with OpenAI/Anthropic API
import anthropic

client = anthropic.Anthropic(api_key="your-key")

# Generate diverse examples
prompts = ["Explain...", "How to...", "What is..."]
# Generate responses and save to JSONL
```

### Method 4: Distillation
Use a stronger model (GPT-4, Claude) to generate responses to your prompts, creating a dataset for a smaller model.

---

## Data Quality Best Practices

1. **Diversity**: Cover various topics, styles, and difficulty levels
2. **Accuracy**: Verify factual correctness
3. **Length**: Mix short and long responses
4. **Format**: Maintain consistent JSON structure
5. **Balance**: For preference data, balance chosen/rejected or true/false

---

## Validation Split

Always create train/validation splits:

```python
from sklearn.model_selection import train_test_split
import jsonlines

# Read data
with jsonlines.open('your_data.jsonl') as reader:
    data = list(reader)

# Split 90/10
train, val = train_test_split(data, test_size=0.1, random_state=42)

# Save splits
with jsonlines.open('train.jsonl', 'w') as writer:
    writer.write_all(train)

with jsonlines.open('val.jsonl', 'w') as writer:
    writer.write_all(val)
```

---

## Recommended Dataset Sizes

| Method | Minimum | Recommended | Large-Scale |
|--------|---------|-------------|-------------|
| SFT    | 1K      | 10K-50K     | 100K+       |
| DPO    | 500     | 5K-20K      | 50K+        |
| KTO    | 1K      | 10K-50K     | 100K+       |

**Note**: Quality > Quantity. 1K high-quality examples often outperform 10K low-quality ones.
