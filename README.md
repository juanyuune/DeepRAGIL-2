# DeepRAGIL-2

Code for the paper: **DeepRAGIL-2: a retrieval-augmented protein language model framework for sensitive and accurate prediction of IL-2-inducing peptides**

Juan Peter Timothy Yuune, Van The Le, Yu-Yen Ou  
Yuan Ze University, Taiwan  
Contact: yien@saturn.yzu.edu.tw

---

## About

This repo contains the code used in our paper. The model combines ESM-2 embeddings, a multi-window CNN, and a RAG module to predict IL-2-inducing peptides. The main motivation was that existing tools (IL2pred, IL2Pepscan) basically predict everything as negative, so we tried to fix that.

Results on independent test set (1:4 class imbalance, 766 sequences):

| Metric | Value |
|--------|-------|
| Sensitivity | 0.7501 |
| Specificity | 0.9708 |
| Accuracy | 0.9281 |
| MCC | 0.7605 |
| AUC | 0.8654 |

---

## Requirements

Tested on Python 3.8, PyTorch 2.0, TensorFlow 2.10, CUDA 11.7.

```bash
pip install -r requirements.txt
```

---

## Data

All sequences are from IEDB (https://www.iedb.org). We used 3,825 sequences after CD-HIT filtering at 40% identity — 738 positive and 3,087 negative, split 80:20.

Sequence IDs with train/test labels are in `data/dataset_ids.csv`. We can't redistribute the raw sequences directly so you'll need to download them from IEDB using the IDs provided.

The RAG database uses a separate set of 3,092 sequences (IDs in `data/rag_database_ids.csv`).

---

## How to run

Run the steps in order. Each step depends on the output of the previous one.

**Step 1 — Generate ESM-2 embeddings**

```bash
python CODE/get_esm2.py -in data/fasta/train/ -out data/embeddings/train/
python CODE/get_esm2.py -in data/fasta/test/ -out data/embeddings/test/
```

**Step 2 — Build the RAG database (only once)**

```bash
python CODE/get_RAGemb.py --model esm2 --fasta data/rag_database.fasta --output data/rag/
```

**Step 3 — Fuse with RAG**

```bash
python CODE/rag_retriever.py --query data/embeddings/train/ --database data/rag/rag_db_esm2.npy --output data/fused/fused_train.npy --query_weight 0.50 --maxseq 35 --emb_dim 1280 --metric cosine

python CODE/rag_retriever.py --query data/embeddings/test/ --database data/rag/rag_db_esm2.npy --output data/fused/fused_test.npy --query_weight 0.50 --maxseq 35 --emb_dim 1280 --metric cosine
```

**Step 4 — Train and evaluate**

```bash
python CODE/MCNN.py --train_data data/fused/fused_train.npy --test_data data/fused/fused_test.npy --train_labels data/labels_train.npy --test_labels data/labels_test.npy --output results/ --window_sizes 8 16 --filters 256 --hidden 500 --epochs 20 --batch_size 512 --mode independent
```

---

## Files

```
CODE/
├── MCNN.py           # model definition and training
├── rag_retriever.py  # RAG fusion
├── get_RAGemb.py     # build RAG database embeddings
├── get_esm2.py       # ESM-2 embeddings
├── get_prottrans.py  # ProtTrans embeddings (used for comparison)
├── get_ankh.py       # Ankh embeddings (used for comparison)
└── import_data.py    # data loading

data/
├── dataset_ids.csv       # sequence IDs with labels and split
└── rag_database_ids.csv  # RAG database sequence IDs
```

---

## Pre-trained weights

Model weights (.h5) available here: [Google Drive — https://drive.google.com/file/d/1c0X9uXocyUp2hq-rvJP8Wqx9EFJEwJvU/view?usp=sharing]

---

## Citation

Paper under review. Will update once published.
