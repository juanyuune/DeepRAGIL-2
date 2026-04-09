# DeepRAGIL-2

A retrieval-augmented protein language model framework for sensitive 
and accurate prediction of IL-2-inducing peptides.

## Publication

Yuune, J. P. T., Le, V. T. & Ou, Y.-Y. DeepRAGIL-2: a 
retrieval-augmented protein language model framework for sensitive 
and accurate prediction of IL-2-inducing peptides. 
*Scientific Reports* (2025). [Link will be added upon acceptance]

## Performance on Independent Test Set

| Metric      | Value  |
|-------------|--------|
| Sensitivity | 0.7501 |
| Specificity | 0.9708 |
| Accuracy    | 0.9281 |
| MCC         | 0.7605 |
| AUC         | 0.8654 |

## Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA 11.7 or higher (recommended for GPU)

## Installation

```bash
git clone https://github.com/[yourusername]/DeepRAGIL-2
cd DeepRAGIL-2
pip install -r requirements.txt
```

## Quick Start — Predict on New Sequences

```bash
python predict/predict.py --input data/sample_sequences.fasta --output results.csv
```

## Dataset

All sequences were sourced from the Immune Epitope Database (IEDB).
Download instructions are in `data/README_data.md`.
A small sample file `data/sample_sequences.fasta` is included 
for testing.

## Pre-trained Model Weights

The best-performing PKL model file is available for download at:
[Google Drive link — add before submission]

Best model: ESM2 + RAG + MCNN windows 8,16
AUC = 0.8654

## Repository Structure
