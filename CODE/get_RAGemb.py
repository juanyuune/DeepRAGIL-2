import os
import gc
import sys
import logging
import argparse
import numpy as np
import torch
from Bio import SeqIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LENGTH = 35


def load_model(model_name):
    logging.info(f"Loading {model_name} model...")
    if model_name == "esm2":
        import esm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        tokenizer = alphabet.get_batch_converter()
        model = model.to(device).eval()
        dim = 1280

    elif model_name == "prottrans":
        from transformers import T5Tokenizer, T5EncoderModel
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
        # Rostlab/prot_t5_xl_uniref50
        model = model.to(device).eval()
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
        dim = 1024

    elif model_name == "ankh":
        import ankh
        model, tokenizer = ankh.load_large_model()
        model = model.to(device).eval()
        dim = 1536

    logging.info(f"Model loaded on {device}.")
    return model, tokenizer, dim


def get_embedding(model, tokenizer, seq, model_name, dim):
    L = len(seq)
    try:
        with torch.no_grad():
            if model_name == "esm2":
                _, _, tokens = tokenizer([("query", seq)])
                out = model(tokens.to(device), repr_layers=[33])["representations"][33][0, 1:L+1, :].cpu().numpy()

            elif model_name == "prottrans":
                inputs = tokenizer(" ".join(list(seq)), return_tensors="pt")
                out = model(input_ids=inputs.input_ids.to(device)).last_hidden_state[0, :L, :].cpu().numpy()

            elif model_name == "ankh":
                outputs = tokenizer.batch_encode_plus(
                    [list(seq)],
                    add_special_tokens=True,
                    padding=True,
                    is_split_into_words=True,
                    return_tensors="pt"
                )
                input_ids      = outputs["input_ids"].to(device)
                attention_mask = outputs["attention_mask"].to(device)
                emb_out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                out = emb_out.last_hidden_state[0, 1:L+1].cpu().numpy()
                del input_ids, attention_mask, emb_out

        # pad to fixed length
        p = np.zeros((MAX_SEQ_LENGTH, dim), dtype=np.float32)
        p[:L, :] = out
        return p

    except RuntimeError as e:
        logging.error(f"RuntimeError: {e}")
        return None
    finally:
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, required=True, choices=["esm2", "prottrans", "ankh"])
    parser.add_argument("--fasta",  type=str, required=True, help="path to RAG database FASTA file")
    parser.add_argument("--output", type=str, required=True, help="output folder")
    args = parser.parse_args()

    if not os.path.exists(args.fasta):
        raise FileNotFoundError(f"FASTA not found: {args.fasta}")
    os.makedirs(args.output, exist_ok=True)

    model, tokenizer, dim = load_model(args.model)

    records = list(SeqIO.parse(args.fasta, "fasta"))
    logging.info(f"Found {len(records)} sequences in {args.fasta}")

    db_data = []
    for i, record in enumerate(records, 1):
        seq = str(record.seq)[:MAX_SEQ_LENGTH]
        if len(seq) == 0:
            logging.warning(f"Skipping empty sequence: {record.id}")
            continue

        emb = get_embedding(model, tokenizer, seq, args.model, dim)
        if emb is not None:
            db_data.append(emb)

        if i % 100 == 0:
            logging.info(f"[{i}/{len(records)}] processed")

    if not db_data:
        logging.error("No embeddings generated. Check your FASTA file.")
        sys.exit(1)

    final_arr   = np.array(db_data).reshape(len(db_data), 1, MAX_SEQ_LENGTH, dim)
    output_path = os.path.join(args.output, f"rag_db_{args.model}.npy")
    np.save(output_path, final_arr)
    logging.info(f"Saved: {output_path}  shape={final_arr.shape}  entries={final_arr.shape[0]}")
