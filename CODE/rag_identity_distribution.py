import os
import csv
import numpy as np

# paths
POS_TEST_NPY = r"C:\jupyter\juan\IL2\dataset\esm2\max_35\pos_real_test.npy"
NEG_TEST_NPY = r"C:\jupyter\juan\IL2\dataset\esm2\max_35\neg_real_test.npy"
RAG_DB_NPY   = r"C:\jupyter\juan\IL2\data\rag\max_35\rag_db_esm2.npy"

TOP_K      = 5
OUTPUT_DIR = "rag_identity_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalise(arr):
    n    = arr.shape[0]
    flat = arr.reshape(n, -1).astype(np.float32)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-10
    return flat / norms


def get_topk_sims(query_norm, db_norm, top_k):
    sims      = query_norm @ db_norm.T
    top_k_idx = np.argsort(sims, axis=1)[:, -top_k:][:, ::-1]
    return np.take_along_axis(sims, top_k_idx, axis=1)


def summary_stats(sims):
    return {
        "n_pairs":        len(sims),
        "min":            float(np.min(sims)),
        "max":            float(np.max(sims)),
        "mean":           float(np.mean(sims)),
        "median":         float(np.median(sims)),
        "std":            float(np.std(sims)),
        "pct_above_0.90": float((sims > 0.90).mean()),
        "pct_above_0.95": float((sims > 0.95).mean()),
        "pct_above_0.99": float((sims > 0.99).mean()),
    }


def main():
    print("Loading embeddings...")
    pos_test = np.load(POS_TEST_NPY).astype(np.float32)
    neg_test = np.load(NEG_TEST_NPY).astype(np.float32)
    rag_db   = np.load(RAG_DB_NPY).astype(np.float32)

    n_pos = len(pos_test)
    n_neg = len(neg_test)
    test_all = np.concatenate([pos_test, neg_test], axis=0)
    print(f"  test: {test_all.shape}  rag_db: {rag_db.shape}")

    print("Computing cosine similarities...")
    test_norm = normalise(test_all)
    db_norm   = normalise(rag_db)
    topk_sims = get_topk_sims(test_norm, db_norm, TOP_K)
    print(f"  top-{TOP_K} sims shape: {topk_sims.shape}")

    sims_all = topk_sims.flatten()
    sims_pos = topk_sims[:n_pos].flatten()
    sims_neg = topk_sims[n_pos:].flatten()

    stats_all = summary_stats(sims_all)
    stats_pos = summary_stats(sims_pos)
    stats_neg = summary_stats(sims_neg)

    print(f"\nAll queries:  mean={stats_all['mean']:.4f}  "
          f"max={stats_all['max']:.4f}  >0.99={stats_all['pct_above_0.99']*100:.2f}%")
    print(f"Positive:     mean={stats_pos['mean']:.4f}  "
          f"max={stats_pos['max']:.4f}  >0.99={stats_pos['pct_above_0.99']*100:.2f}%")
    print(f"Negative:     mean={stats_neg['mean']:.4f}  "
          f"max={stats_neg['max']:.4f}  >0.99={stats_neg['pct_above_0.99']*100:.2f}%")

    # save summary stats
    stats_csv = os.path.join(OUTPUT_DIR, "similarity_summary_stats.csv")
    with open(stats_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "n_pairs", "min", "max", "mean", "median",
                         "std", "pct_above_0.90", "pct_above_0.95", "pct_above_0.99"])
        for group, st in [("all", stats_all), ("positive", stats_pos), ("negative", stats_neg)]:
            writer.writerow([
                group,
                st["n_pairs"],
                round(st["min"],    4),
                round(st["max"],    4),
                round(st["mean"],   4),
                round(st["median"], 4),
                round(st["std"],    4),
                round(st["pct_above_0.90"], 4),
                round(st["pct_above_0.95"], 4),
                round(st["pct_above_0.99"], 4),
            ])
    print(f"\nSaved: {stats_csv}")

    # save per-query results
    perquery_csv = os.path.join(OUTPUT_DIR, "per_query_topk_similarities.csv")
    labels = ["positive"] * n_pos + ["negative"] * n_neg
    with open(perquery_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_index", "class"] + [f"sim_rank{k+1}" for k in range(TOP_K)])
        for i in range(len(topk_sims)):
            writer.writerow([i, labels[i]] + [round(float(topk_sims[i, k]), 4)
                                              for k in range(TOP_K)])
    print(f"Saved: {perquery_csv}")


if __name__ == "__main__":
    main()
