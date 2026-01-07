# Low-Latency-Prompt-Optimization-with-Structured-Complementary-Prompt

[![EACL 2026](https://img.shields.io/badge/EACL-2026-blue)](https://2026.eacl.org/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

This repository contains the official implementation of the paper "Don’t Generate, Classify! Low-Latency Prompt Optimization with Structured Complementary Prompt" published at **EACL 2026** (19th Conference of the European Chapter of the Association for Computational Linguistics).

## 📋 Overview

This project implements LLPO (Low Latency Prompt Optimization), a novel approach presented at EACL 2026 that uses classification-based methods instead of generation-based approaches to optimize prompts for language models. By using a classifier to identify the appropriate domain/category and then selecting pre-optimized prompts, LLPO achieves significantly lower latency compared to generation-based prompt optimization methods.

The main components include:

- **Field Clustering**: Semantically clustering diverse domain labels into similar groups to reduce classification complexity
- **Multi-task Classifier**: Training efficient classification models using clustered labels for fast inference
- **Prompt Optimization**: Comparing LLPO with other optimization methods including batch-based and field-wise approaches

## 🏗️ Project Structure

```
LLPO_github/
├── clustering/                          # Field clustering module
│   ├── fields_clustering_kmeans.py      # K-means clustering
│   ├── fields_clustering_agglomerative.py  # Hierarchical clustering
│   ├── functions.py                     # Common utility functions
│   └── sh/                              # Execution scripts
├── classifier/                          # Classifier training module
│   ├── train.py                         # Multi-task classifier training
│   └── sh/                              # Model-specific training scripts
│       ├── RoBERTa_kmeans.sh
│       ├── RoBERTa_agg.sh
│       ├── DeBERTa_kmeans.sh
│       ├── DeBERTa_agg.sh
│       ├── ModernBERT_kmeans.sh
│       └── ModernBERT_agg.sh
├── evaluation/                          # Evaluation module
│   ├── optimization/                    # Optimization algorithms
│   │   ├── LLPO_optimization.py         # Low Latency Prompt Optimization
│   │   ├── BPO_optimization.py          # Batch Prompt Optimization
│   │   ├── PAS_optimization.py          # Prompt Augmentation Strategy
│   │   └── FIPO_optimization.py         # Field-wise Instruction Prompt Optimization
│   ├── inference/                       # Inference scripts
│   │   ├── inference_OpenAI.py
│   │   ├── inference_Anthropic.py
│   │   └── inference_opensource.py
│   ├── scoring/                         # Evaluation metrics
│   │   ├── scoring_gpt.py
│   │   └── win_tie_lose.py
│   └── sh/                              # Execution scripts
│       ├── optimization/
│       └── inference/
└── prompts/                           


```

## 🛠️ Environment Setup

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Field Clustering

Cluster diverse domain labels into semantically similar groups.

#### K-means Clustering
```bash
python clustering/fields_clustering_kmeans.py
```

#### Hierarchical Clustering (Agglomerative)
```bash
python clustering/fields_clustering_agglomerative.py
```

**Output Files**:
- `field_clusters_{method}.json`: Clustering results
- `lookup_table_{method}.json`: Original label → Representative label mapping
- `replaced_label_data_{method}.json`: Dataset with clustered labels

### 2. Classifier Training

Train multi-task classifiers using the clustered labels.

#### RoBERTa Model (K-means Clusters)
```bash
bash classifier/sh/RoBERTa_kmeans.sh
```

#### DeBERTa Model (Agglomerative Clusters)
```bash
bash classifier/sh/DeBERTa_agg.sh
```

#### ModernBERT Model
```bash
bash classifier/sh/ModernBERT_kmeans.sh
bash classifier/sh/ModernBERT_agg.sh
```

**Key Hyperparameters**:
- `--model_name`: Pre-trained model (RoBERTa, DeBERTa, ModernBERT, etc.)
- `--batch_size`: Batch size
- `--accumulation_steps`: Gradient accumulation steps
- `--learning_rate`: Learning rate
- `--dropout_rate`: Dropout rate
- `--gamma`: Gamma parameter for Focal Loss

### 3. Prompt Optimization and Evaluation

#### LLPO (Low Latency Prompt Optimization)
```bash
bash evaluation/sh/optimization/optimized_LLPO.sh
```

#### Other Prompt Optimization Methods
```bash
bash evaluation/sh/optimization/optimized_BPO.sh    # Batch Prompt Optimization
bash evaluation/sh/optimization/optimized_PAS.sh    # Prompt Augmentation Strategy
bash evaluation/sh/optimization/optimized_FIPO.sh   # Field-wise Instruction Prompt Optimization
```

#### Inference and Evaluation
```bash
bash evaluation/sh/inference/inference_LLPO.sh
bash evaluation/sh/inference/inference_BPO.sh
```

## 📊 Datasets

### SCP Dataset
- `SCP.json`: Main dataset with instruction examples
- `minilm_field_embeddings.pkl`: Field embeddings (using MiniLM model)
- `prompts/`: Prompt templates for different optimization methods
  - `BPO_prompt.txt`: Template for Batch Prompt Optimization
  - `PAS_prompt.txt`: Template for Prompt Augmentation Strategy

### Cluster Data
- `final_kmeans/`: K-means clustering results
- `final_agglomerative/`: Hierarchical clustering results

Each folder contains:
- `field_clusters_*.json`: Label groups by cluster
- `lookup_table_*.json`: Label conversion table
- `replaced_label_data_*.json`: Training data with clustered labels applied


## 📈 Evaluation

### Win/Tie/Lose Analysis
```bash
python evaluation/scoring/win_tie_lose.py
```

### GPT-based Scoring
```bash
python evaluation/scoring/scoring_gpt.py
```

## 🛠️ Supported Models

- **RoBERTa**: `FacebookAI/roberta-large`
- **DeBERTa**: `microsoft/deberta-v3-large`
- **ModernBERT**: `answerdotai/ModernBERT-large`
- Other HuggingFace Transformers compatible models

## 📝 Citation

If you use this code, please cite:

```bibtex
@inproceedings{dont-generate-classify-2026,
  title={Don't Generate, Classify: Low Latency Prompt Optimization},
  author={[Authors]},
  booktitle={Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  year={2026}
}
```
