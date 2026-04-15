import os
import gc
import pickle
import logging
import argparse
import torch
import ankh

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_ankh(model_size="large"):
    logging.info(f"Loading Ankh-{model_size} model...")
    if model_size == "large":
        model, tokenizer = ankh.load_large_model()
    else:
        model, tokenizer = ankh.load_base_model()
    model = model.to(device).eval()
    return model, tokenizer

def read_fasta(fasta_path):
    seq = ''
    with open(fasta_path, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                seq += line.strip()
    seq_id = os.path.splitext(os.path.basename(fasta_path))[0]
    return [(seq_id, seq)]

def get_embeddings(model, tokenizer, seqs):
    results = {}
    for sid, seq in seqs:
        outputs = tokenizer.batch_encode_plus(
            [list(seq)],
            add_special_tokens=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt"
        )
        input_ids      = outputs["input_ids"].to(device)
        attention_mask = outputs["attention_mask"].to(device)
        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # strip BOS and EOS tokens
            emb = out.last_hidden_state[0, 1:len(seq)+1].detach().cpu().numpy()
            results[sid] = emb
            del out
        except RuntimeError as e:
            logging.error(f"RuntimeError during embedding: {e}")
        finally:
            del input_ids, attention_mask
            torch.cuda.empty_cache()
            gc.collect()
    return results

def save_embeddings(data, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

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
    parser.add_argument("-in",    "--path_input",  type=str, required=True)
    parser.add_argument("-out",   "--path_output", type=str, required=True)
    parser.add_argument("-model", "--model_size",  type=str, default="large", choices=["base", "large"])
    args = parser.parse_args()

    if not os.path.exists(args.path_input):
        raise FileNotFoundError(f"Input path not found: {args.path_input}")
    os.makedirs(args.path_output, exist_ok=True)

    model, tokenizer = load_ankh(args.model_size)

    fasta_files = [f for f in os.listdir(args.path_input) if f.endswith(".fasta")]
    for i, fname in enumerate(fasta_files, 1):
        in_path  = os.path.join(args.path_input, fname)
        out_path = os.path.join(args.path_output, os.path.splitext(fname)[0] + ".ankh")
        logging.info(f"[{i}/{len(fasta_files)}] {fname}")
        process_fasta(in_path, out_path, model, tokenizer)
        gc.collect()

    logging.info(f"Finished. {len(fasta_files)} files processed.")
    
