import os
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_db_metadata(meta_path, n_entries):
    # load optional CSV metadata, fall back to synthetic IDs if not provided
    if meta_path and Path(meta_path).exists():
        df = pd.read_csv(meta_path)
        if len(df) != n_entries:
            logging.warning(
                f"metadata rows ({len(df)}) != database entries ({n_entries}), "
                f"using synthetic IDs instead"
            )
            df = pd.DataFrame({"id": [f"Protein_{i}" for i in range(n_entries)]})
    else:
        df = pd.DataFrame({"id": [f"Protein_{i}" for i in range(n_entries)]})
    return df


def load_npy(path, maxseq, emb_dim):
    # load .npy and reshape to (N, maxseq, emb_dim)
    data = np.load(path)
    if data.ndim == 4:
        data = data.reshape(data.shape[0], maxseq, emb_dim)
    elif data.ndim == 2:
        data = data[:, np.newaxis, :]
        data = np.repeat(data, maxseq, axis=1)
    elif data.ndim != 3:
        raise ValueError(f"unexpected embedding shape: {data.shape}")
    return data.astype(np.float32)


class RAGRetriever:

    def __init__(self, database_path, db_meta_path=None,
                 top_k=5, query_weight=0.7,
                 maxseq=1000, emb_dim=1024, metric="l2"):

        self.top_k        = top_k
        self.query_weight = query_weight
        self.rag_weight   = 1.0 - query_weight
        self.maxseq       = maxseq
        self.emb_dim      = emb_dim
        self.metric       = metric

        logging.info(f"loading RAG database from {database_path}")
        self.database = load_npy(database_path, maxseq, emb_dim)
        logging.info(f"database shape: {self.database.shape}")

        self.metadata = load_db_metadata(db_meta_path, self.database.shape[0])

        if metric == "cosine":
            flat = self.database.reshape(self.database.shape[0], -1)
            self._db_norms        = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-10
            self._db_flat_normed  = flat / self._db_norms

    def _distances(self, query_flat):
        if self.metric == "cosine":
            q_norm = query_flat / (np.linalg.norm(query_flat) + 1e-10)
            return 1.0 - (self._db_flat_normed @ q_norm)
        else:
            db_flat = self.database.reshape(self.database.shape[0], -1)
            return np.linalg.norm(db_flat - query_flat, axis=1)

    def retrieve(self, query, return_metadata=False):
        # normalise query to (maxseq, emb_dim)
        q = query.squeeze()
        if q.ndim == 1:
            q = np.tile(q[np.newaxis, :], (self.maxseq, 1))
        elif q.shape != (self.maxseq, self.emb_dim):
            raise ValueError(
                f"query shape {q.shape} does not match "
                f"expected ({self.maxseq}, {self.emb_dim})"
            )

        query_flat  = q.reshape(-1).astype(np.float32)
        dists       = self._distances(query_flat)
        top_k_idx   = np.argsort(dists)[:self.top_k]
        top_k_dists = dists[top_k_idx]

        # softmax-style weights from inverse distances
        sim_scores = 1.0 / (top_k_dists + 1e-10)
        weights    = sim_scores / sim_scores.sum()

        retrieved = self.database[top_k_idx]
        rag_embed = np.average(retrieved, axis=0, weights=weights)
        fused     = self.query_weight * q + self.rag_weight * rag_embed

        if return_metadata:
            meta = self.metadata.iloc[top_k_idx].copy()
            meta["distance"] = top_k_dists
            meta["weight"]   = weights
            return fused, meta

        return fused

    def retrieve_batch(self, queries, batch_size=100):
        if queries.ndim == 4:
            queries = queries.reshape(queries.shape[0], self.maxseq, self.emb_dim)

        n         = queries.shape[0]
        fused     = np.empty_like(queries)
        n_batches = (n + batch_size - 1) // batch_size

        for b in range(n_batches):
            start = b * batch_size
            end   = min(start + batch_size, n)
            for i in range(start, end):
                fused[i] = self.retrieve(queries[i])
            logging.info(f"batch {b+1}/{n_batches} done ({start}-{end-1})")

        return fused

    def discriminate(self, embeddings, threshold=0.5, return_scores=False):
        if embeddings.ndim == 4:
            embeddings = embeddings.reshape(
                embeddings.shape[0], self.maxseq, self.emb_dim
            )

        scores = np.array([
            self._distances(embeddings[i].reshape(-1)).min()
            for i in range(embeddings.shape[0])
        ])
        mask = scores <= threshold

        logging.info(
            f"{mask.sum()}/{len(mask)} embeddings passed "
            f"(threshold={threshold:.3f})"
        )

        if return_scores:
            return mask, scores
        return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",          required=True)
    parser.add_argument("--database",       required=True)
    parser.add_argument("--db_meta",        default=None)
    parser.add_argument("--output",         required=True)
    parser.add_argument("--top_k",          type=int,   default=5)
    parser.add_argument("--query_weight",   type=float, default=0.7)
    parser.add_argument("--batch_size",     type=int,   default=100)
    parser.add_argument("--maxseq",         type=int,   default=1000)
    parser.add_argument("--emb_dim",        type=int,   default=1024)
    parser.add_argument("--metric",         default="l2", choices=["l2", "cosine"])
    parser.add_argument("--discriminate",   action="store_true")
    parser.add_argument("--disc_threshold", type=float, default=0.5)
    args = parser.parse_args()

    rag = RAGRetriever(
        database_path=args.database,
        db_meta_path=args.db_meta,
        top_k=args.top_k,
        query_weight=args.query_weight,
        maxseq=args.maxseq,
        emb_dim=args.emb_dim,
        metric=args.metric,
    )

    logging.info(f"loading queries from {args.query}")
    queries = load_npy(args.query, args.maxseq, args.emb_dim)
    logging.info(f"query shape: {queries.shape}")

    if args.discriminate:
        mask, scores = rag.discriminate(
            queries, threshold=args.disc_threshold, return_scores=True
        )
        logging.info(f"first 10 scores: {scores[:10].round(4)}")
        filtered = queries[mask]
        out_path = Path(args.output)
        np.save(
            str(out_path.with_stem(out_path.stem + "_filtered")),
            filtered.reshape(filtered.shape[0], 1, args.maxseq, args.emb_dim)
        )
        logging.info(f"saved filtered queries: {mask.sum()}/{len(mask)} passed")
    else:
        fused    = rag.retrieve_batch(queries, batch_size=args.batch_size)
        fused_4d = fused.reshape(fused.shape[0], 1, args.maxseq, args.emb_dim)
        np.save(args.output, fused_4d)
        logging.info(f"saved fused embeddings to {args.output}, shape {fused_4d.shape}")


if __name__ == "__main__":
    main()
