# LLM Post-Training Framework

A minimal framework for post-training LLMs using various alignment methods.

## Structure

```
.
├── configs/           # YAML configuration files
├── data/              # Training and evaluation data
├── scripts/           # Training scripts (one per method)
├── evaluate.py        # Evaluation and plotting
├── outputs/           # Trained models
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# SFT
python scripts/train_sft.py --config configs/sft.yaml

# DPO
python scripts/train_dpo.py --config configs/dpo.yaml

# ORPO
python scripts/train_orpo.py --config configs/orpo.yaml

# KTO
python scripts/train_kto.py --config configs/kto.yaml

# GRPO
python scripts/train_grpo.py --config configs/grpo.yaml

# Nash-MD
python scripts/train_nash.py --config configs/nash.yaml

# Evaluate
python evaluate.py --model outputs/sft --data data/eval.jsonl
```

---

## Methods

### SFT (Supervised Fine-Tuning)

**What:** Train model to imitate desired outputs given instructions.

**Data format:**
```json
{"instruction": "What is 2+2?", "output": "4"}
```

**When to use:** First stage of training. Teaches the model to follow instructions and generate properly formatted responses.

**Pros:** Simple, stable, efficient
**Cons:** Only learns to imitate, doesn't learn preferences

---

### DPO (Direct Preference Optimization)

**What:** Align model using pairs of preferred vs rejected responses, without RL.

**Data format:**
```json
{"prompt": "Explain AI", "chosen": "AI is...", "rejected": "idk lol"}
```

**When to use:** After SFT. When you have human preference data (A is better than B).

**How it works:** Increases probability of chosen responses, decreases probability of rejected ones, relative to a frozen reference model.

**Pros:** No reward model needed, stable training, offline
**Cons:** Requires paired preference data

---

### ORPO (Odds Ratio Preference Optimization)

**What:** Combines SFT and preference learning in a single stage.

**Data format:** Same as DPO (prompt, chosen, rejected)

**When to use:** When you want to skip the SFT stage and do alignment directly from base model.

**How it works:** Uses odds ratio of generating chosen vs rejected to create a preference signal, combined with SFT loss.

**Pros:** Single stage (faster), no reference model needed
**Cons:** May be less stable than SFT→DPO pipeline

---

### KTO (Kahneman-Tversky Optimization)

**What:** Align using binary feedback (good/bad) instead of pairwise comparisons.

**Data format:**
```json
{"prompt": "Write code", "completion": "def foo()...", "label": true}
{"prompt": "Write code", "completion": "idk", "label": false}
```

**When to use:** When you have thumbs up/down feedback instead of A vs B comparisons.

**How it works:** Based on prospect theory from behavioral economics. Treats gains (good responses) and losses (bad responses) asymmetrically.

**Pros:** Easier data collection (binary labels), works with unpaired data
**Cons:** Less signal per example than pairwise

---

### GRPO (Group Relative Policy Optimization)

**What:** Online RL method that generates multiple completions and uses reward scores.

**Data format:**
```json
{"prompt": "Explain quantum computing"}
```

**When to use:** When you have a reward model and want online optimization.

**How it works:**
1. Generate N completions per prompt
2. Score with reward model
3. Use relative rewards within group for policy gradient

**Pros:** Uses reward signal directly, online learning
**Cons:** Requires reward model, more compute (generation during training)

---

### Nash-MD (Nash Learning from Human Feedback)

**What:** Game-theoretic approach that finds a Nash equilibrium between policy and preference model.

**Data format:**
```json
{"prompt": "Explain quantum computing"}
```

**When to use:**
- When you want theoretically grounded alignment with equilibrium guarantees
- When dealing with multiple competing objectives
- For robust alignment that avoids mode collapse

**How it works:**
1. Treats alignment as a two-player game
2. Policy tries to maximize reward, preference model tries to distinguish good/bad
3. Converges to Nash equilibrium where neither can improve unilaterally
4. Uses mixture of policies to ensure exploration

**Pros:** Strong theoretical guarantees, avoids reward hacking, robust
**Cons:** More complex, requires reward model/judge, online method

---

## Comparison

| Method | Data Type | Stages | Online/Offline | Needs Reward Model |
|--------|-----------|--------|----------------|-------------------|
| SFT | instruction→output | 1 | Offline | No |
| DPO | chosen vs rejected | 2 (after SFT) | Offline | No |
| ORPO | chosen vs rejected | 1 | Offline | No |
| KTO | binary labels | 2 (after SFT) | Offline | No |
| GRPO | prompts only | 2 (after SFT) | Online | Yes |
| Nash-MD | prompts only | 2 (after SFT) | Online | Yes |

---

## LoRA / QLoRA

All configs support:

- `use_lora: true/false` — Enable LoRA adapters
- `use_qlora: true/false` — Enable 4-bit quantization

**Full fine-tuning:** Set both to `false` (requires more VRAM)

**QLoRA (recommended):** Set both to `true` (works on 12-16GB GPUs)

---

## Data Format

The `data/` directory should contain JSONL files with examples.

**For SFT:**
```json
{"instruction": "...", "output": "..."}
```

**For DPO/ORPO:**
```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

**For KTO:**
```json
{"prompt": "...", "completion": "...", "label": true/false}
```

**For GRPO/Nash-MD:**
```json
{"prompt": "..."}
```

The provided `data/train.jsonl` contains all fields for compatibility with any method.

---

## Evaluation

```bash
# Single model
python evaluate.py --model outputs/sft --data data/eval.jsonl

# Compare models
python evaluate.py --model outputs/sft --data data/eval.jsonl --compare outputs/dpo outputs/orpo
```

Generates:
- `training_curves.png` — Loss over time
- `perplexity_distribution.png` — Model perplexity histogram
- `response_lengths.png` — Generated response lengths
- `model_comparison.png` — Side-by-side metrics
- `evaluation_stats.json` — Numeric results
- `sample_responses.json` — Example generations

---

## Hardware Requirements

| Setup | VRAM | Config |
|-------|------|--------|
| QLoRA | 12-16 GB | `use_qlora: true` |
| LoRA | 24-40 GB | `use_lora: true, use_qlora: false` |
| Full | 40-80 GB | `use_lora: false, use_qlora: false` |

---

## References

- [TRL Documentation](https://huggingface.co/docs/trl)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [ORPO Paper](https://arxiv.org/abs/2403.07691)
- [KTO Paper](https://arxiv.org/abs/2402.01306)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [Nash-MD Paper](https://arxiv.org/abs/2312.00886)
