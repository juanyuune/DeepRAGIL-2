# DeepRAGIL-2

A retrieval-augmented protein language model framework for sensitive and accurate prediction of IL-2-inducing peptides.

**Authors:** Juan Peter Timothy Yuune, Van The Le, Yu-Yen Ou  
**Affiliation:** Department of Computer Science and Engineering, Yuan Ze University, Taiwan  
**Contact:** yien@saturn.yzu.edu.tw

---

## Overview

DeepRAGIL-2 combines three components for IL-2 induction prediction:

- **ESM-2** embeddings (esm2_t33_650M_UR50D)
- **Dual-window MCNN** with kernel sizes 8 and 16
- **RAG module** with equal fusion weighting (query = 0.50, RAG = 0.50)

---

## Performance

| Metric | Value |
|--------|-------|
| Sensitivity | 0.7501 |
| Specificity | 0.9708 |
| Accuracy | 0.9281 |
| MCC | 0.7605 |
| AUC | 0.8654 |

---

## Requirements

```
pip install -r requirements.txt
```

---

## Dataset

Sequences were sourced from IEDB (https://www.iedb.org). The dataset contains 3,825 non-redundant sequences (738 IL-2-inducing, 3,087 non-inducing) split 80:20 into training and independent test sets.

---

## Usage

**Step 1 — Generate ESM-2 embeddings**
```bash
python CODE/get_esm2.py -in data/fasta/ -out data/embeddings/esm2/
```

**Step 2 — Build RAG database**
```bash
python CODE/get_RAGemb.py -in data/rag_fasta/ -out data/embeddings/rag/
```

**Step 3 — Prepare dataset**
```bash
python CODE/get_datasets.py --embeddings data/embeddings/esm2/ --labels data/dataset_ids.csv --output data/processed/
```

**Step 4 — Train and evaluate**
```bash
python CODE/MCNN.py --data data/processed/ --rag_database data/embeddings/rag/ --rag_weight 0.50 --window_sizes 8 16 --filters 256 --hidden 500 --output results/
```

---

## Pre-trained Model Weights

Available at: [(https://drive.google.com/file/d/1c0X9uXocyUp2hq-rvJP8Wqx9EFJEwJvU/view?usp=sharing)]

---

## Citation

Manuscript in preparation (2026). Citation will be updated upon acceptance.
