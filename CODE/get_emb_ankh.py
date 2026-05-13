"""
get_emb_ankh.py
Ankh per-residue embeddings for AIP sequences.

Reads individual FASTA files, generates Ankh encoder
hidden states, and saves each as a .ankh pickle file.
Supports both ankh-base (768d) and ankh-large (1536d).

Usage:
    python get_emb_ankh.py \
        -in  C:/jupyter/juan/AIP/data/data_fasta/pos_train \
        -out C:/jupyter/juan/AIP/data/ankh_embeddings/pos_train \
        -model large

Requirements:
    pip install ankh torch
"""

import os
import gc
import pickle
import logging
import argparse

import numpy as np
import torch
import ankh


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("-in",    "--path_input",  required=True,
                    help="folder of individual .fasta files")
parser.add_argument("-out",   "--path_output", required=True,
                    help="folder to save .ankh embedding files")
parser.add_argument("-model", "--model_size",  default="large",
                    choices=["base", "large"],
                    help="ankh-base (768d) or ankh-large (1536d), default: large")


# ── model ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(size):
    log.info("loading Ankh-%s...", size)
    if size == "large":
        model, tokenizer = ankh.load_large_model()
    else:
        model, tokenizer = ankh.load_base_model()
    model = model.to(device).eval()
    log.info("model ready on %s", device)
    return model, tokenizer


# ── helpers ───────────────────────────────────────────────────────────────────

def read_fasta(path):
    seq = ""
    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                seq += line.strip()
    seq_id = os.path.splitext(os.path.basename(path))[0]
    return [(seq_id, seq)]


def get_embedding(model, tokenizer, seqs):
    """
    Returns {seq_id: np.array of shape (L, hidden_dim)}.
    Strips BOS and EOS tokens to match ESM-2 convention
    of returning one vector per input residue.
    hidden_dim = 768 for ankh-base, 1536 for ankh-large.
    """
    results = {}

    for seq_id, seq in seqs:
        tokens = tokenizer.batch_encode_plus(
            [list(seq)],
            add_special_tokens=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt"
        )
        input_ids      = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True)

        # strip BOS at pos 0 and EOS at pos L+1
        L   = len(seq)
        emb = out.last_hidden_state[0, 1:L+1].detach().cpu().numpy()
        results[seq_id] = emb

        del input_ids, attention_mask, out
        torch.cuda.empty_cache()
        gc.collect()

    return results


def save_embedding(emb, path):
    with open(path, "wb") as f:
        pickle.dump(emb, f)


def log_failed(in_path, out_path):
    with open("NO_OK.txt", "a") as f:
        f.write(f"{in_path} > {out_path}\n")


def process(fasta_path, out_path, model, tokenizer):
    try:
        fname = os.path.splitext(os.path.basename(fasta_path))[0]
        seqs  = read_fasta(fasta_path)
        embs  = get_embedding(model, tokenizer, seqs)
        save_embedding(embs[fname], out_path)
        log.info("saved  %s", os.path.basename(out_path))
    except Exception as e:
        log.error("failed %s: %s", fasta_path, e)
        log_failed(fasta_path, out_path)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.path_input):
        raise FileNotFoundError(f"input not found: {args.path_input}")
    os.makedirs(args.path_output, exist_ok=True)

    model, tokenizer = load_model(args.model_size)

    fasta_files = [f for f in os.listdir(args.path_input)
                   if f.endswith(".fasta")]
    log.info("found %d fasta files in %s", len(fasta_files), args.path_input)

    for i, fname in enumerate(fasta_files, 1):
        in_path  = os.path.join(args.path_input,  fname)
        out_path = os.path.join(args.path_output,
                                os.path.splitext(fname)[0] + ".ankh")
        if i % 100 == 0 or i == 1:
            log.info("[%d/%d] %s", i, len(fasta_files), fname)
        process(in_path, out_path, model, tokenizer)
        gc.collect()

    log.info("done — %d files embedded", len(fasta_files))
