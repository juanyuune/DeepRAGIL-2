import os
import gc
import time
import logging
import argparse
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_prottrans():
    logging.info("Loading ProtTrans model...")
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )
    return model, tokenizer


def read_fasta(fasta_path):
    seq = ''
    with open(fasta_path, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                seq += line.strip()
    seq_id = os.path.splitext(os.path.basename(fasta_path))[0]
    return [(seq_id, seq)]


def get_embeddings(model, tokenizer, seqs, max_residues=4000, max_seq_len=1000, max_batch=100):
    results = {}
    seq_dict = sorted(seqs, key=lambda x: len(x[1]), reverse=True)
    batch = []
    start = time.time()

    for seq_idx, (sid, seq) in enumerate(seq_dict, 1):
        seq_len = len(seq)
        seq_spaced = ' '.join(list(seq))
        batch.append((sid, seq_spaced, seq_len))

        n_res_batch = sum(s_len for _, _, s_len in batch) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            ids, _, lens = zip(*batch)
            seqs_batch = [s for _, s, _ in batch]
            batch = []

            token_encoding = tokenizer.batch_encode_plus(
                seqs_batch, add_special_tokens=True, padding="longest"
            )
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    out = model(input_ids, attention_mask=attention_mask)
            except RuntimeError as e:
                logging.error(f"RuntimeError: {e}")
                continue

            for i, sid in enumerate(ids):
                emb = out.last_hidden_state[i, :lens[i]].detach().cpu().numpy()
                results[sid] = emb

            del input_ids, attention_mask, out
            torch.cuda.empty_cache()

    logging.info(f"Embedded {len(results)} sequences in {time.time() - start:.1f}s")
    return results


def save_embeddings(data, output_path):
    np.savetxt(output_path, data)


def process_fasta(fasta_path, output_path, model, tokenizer):
    try:
        fname = os.path.splitext(os.path.basename(fasta_path))[0]
        seqs = read_fasta(fasta_path)
        results = get_embeddings(model, tokenizer, seqs)
        save_embeddings(results[fname], output_path)
        logging.info(f"Done: {fasta_path} -> {output_path}")
    except Exception as e:
        logging.error(f"Failed on {fasta_path}: {e}")
        with open('NO_OK.txt', 'a') as f:
            f.write(f"{fasta_path} > {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in",  "--path_input",  type=str, required=True)
    parser.add_argument("-out", "--path_output", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.path_input):
        raise FileNotFoundError(f"Input path not found: {args.path_input}")
    os.makedirs(args.path_output, exist_ok=True)

    model, tokenizer = load_prottrans()

    fasta_files = [f for f in os.listdir(args.path_input) if f.endswith(".fasta")]
    for i, fname in enumerate(fasta_files, 1):
        in_path  = os.path.join(args.path_input, fname)
        out_path = os.path.join(args.path_output, os.path.splitext(fname)[0] + ".prottrans")
        logging.info(f"[{i}/{len(fasta_files)}] {fname}")
        process_fasta(in_path, out_path, model, tokenizer)
        gc.collect()

    logging.info(f"Finished. {len(fasta_files)} files processed.")
