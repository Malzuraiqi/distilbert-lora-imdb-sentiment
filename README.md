# DistilBERT LoRA: IMDB Sentiment Analysis

## Overview

Parameter-Efficient Fine-Tuning (PEFT) of a DistilBERT model for sentiment classification using LoRA. By targeting specific attention blocks, the model trains only ~1% of its total parameters, making it lightweight enough to run efficiently on a single free-tier Colab GPU.

## Training Setup

| Setting | Value |
|---|---|
| Hardware | Google Colab T4 GPU (~31 minutes training time) |
| LoRA | `r=8`, `alpha=16`, `dropout=0.1` |
| Learning rate | `1e-3` |
| Epochs | `3` |
| Max length | `256` |
| Seed | `42` |

## Results & Performance

| Metric | Score |
|---|---|
| Test accuracy (full 25k dataset) | **91.2%** |
| Baseline (shuffled 500-sample subset) | **86%** |

## Key Learnings & Debugging

During early testing, the model hit 100% accuracy. The IMDB dataset is sorted by label, so my initial unshuffled 500/200-sample train/test subsets contained only negative reviews. The model simply learned to always predict `negative` and scored perfectly.

Implementing proper dataset shuffling before sampling exposed the actual baseline performance and fixed the training pipeline.

## Known Limitations

While highly accurate on standard reviews, the model struggles with heavy sarcasm. For example:

> "Truly a masterclass in cinematic filmmaking, provided your goal is to make the audience root for the literal meteor to wipe out the entire cast"  
> **Predicted:** Positive  
> **Actual:** Negative

## How to Run

Install the required dependencies:

```bash
pip install torch transformers peft datasets evaluate
```

Load the base model and apply the trained LoRA adapter:

```python
from transformers import AutoModelForSequenceClassification
from peft import PeftModel

# Load the base model
base = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
)

# Load the LoRA adapter (assuming the folder is in your working directory)
model = PeftModel.from_pretrained(base, "./lora_imdb_adapter")
```
