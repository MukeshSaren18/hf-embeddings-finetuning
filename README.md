# Hugging Face — Fine-Tuning & Embeddings for Data Engineering NLP Tasks

![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A practical notebook demonstrating two HuggingFace workflows applied to **real data engineering problems**.

---

## What's Inside

### Part 1 — Semantic Embeddings (`sentence-transformers`)
**Use case:** Schema matching during Hadoop/Hive → Azure Databricks migration.  
Maps legacy column names (`cust_id`, `ord_amt`) to new naming conventions (`customer_id`, `order_amount_eur`) by semantic meaning — handling cases where string matching fails.

Also demonstrates **data catalogue search** — query your Unity Catalog / Purview catalogue by natural language intent.

**Model:** `all-MiniLM-L6-v2` — 22M params, 384-dim embeddings, runs on CPU.

### Part 2 — Fine-Tuning DistilBERT (`transformers` + `Trainer`)
**Use case:** Automated pipeline log classification for the AI-Driven Data Quality Platform.  
Fine-tunes `distilbert-base-uncased` to classify pipeline log messages as:
- `normal` — routine, no action
- `warning` — degraded, monitor
- `error` — failed, alert
- `anomaly` — statistical deviation, data quality investigation

Integrates with MLflow for model tracking and Databricks Model Serving for production deployment.

**Model:** `distilbert-base-uncased` — 66M params, 40% smaller than BERT-base.

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/MukeshSaren18/hf-embeddings-finetuning.git
cd hf-embeddings-finetuning
pip install -r requirements.txt

# Run in Jupyter
jupyter notebook hf_embeddings_finetuning.ipynb
```

**Or open directly in Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

---

## Requirements

```
transformers>=4.40.0
datasets>=2.18.0
sentence-transformers>=3.0.0
torch>=2.2.0
scikit-learn>=1.4.0
evaluate>=0.4.0
accelerate>=0.28.0
matplotlib>=3.8.0
seaborn>=0.13.0
pandas>=2.1.0
numpy>=1.26.0
```

---

## Connection to Production Stack

| Notebook Section | Production Component |
|----------------|---------------------|
| Embeddings — schema matching | Hadoop/Hive → Azure migration tooling |
| Embeddings — catalogue search | Microsoft Purview / Unity Catalog semantic search |
| Fine-tuned classifier — inference | AI-Driven Data Quality Platform (LangChain + DLT) |
| MLflow logging block | Databricks MLflow tracking + Model Registry |
| Model serving | Databricks Model Serving endpoint |

## Tech Stack
`transformers 4.40` · `sentence-transformers 3.0` · `datasets` · `evaluate` · `MLflow` · `scikit-learn` · `PyTorch 2.2`
