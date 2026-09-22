import os
import gc
import csv
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.utils import shuffle

# configuration
MAXSEQ      = 35
NUM_FEATURE = 1280
NUM_FILTER  = 256
NUM_HIDDEN  = 500
BATCH_SIZE  = 512
NUM_CLASSES = 2
EPOCHS      = 20
SEED        = 42

# paths
ORIG_DIR         = r"C:\jupyter\juan\IL2\dataset\esm2\max_35\0.5_rag_fused"
POS_TRAIN        = os.path.join(ORIG_DIR, "pos_real_train_rag.npy")
NEG_TRAIN        = os.path.join(ORIG_DIR, "neg_real_train_rag.npy")
POS_TEST         = os.path.join(ORIG_DIR, "pos_real_test_rag.npy")
NEG_TEST         = os.path.join(ORIG_DIR, "neg_real_test_rag.npy")
NEG_IL2_CSV      = r"C:\jupyter\juan\IL2\iedb_csv\neg\neg_IL2_labeled.csv"
NEG_CYTOKINE_CSV = r"C:\jupyter\juan\IL2\iedb_csv\neg\neg_cytokine_labeled.csv"
NEG_TEST_DIR     = r"C:\jupyter\juan\IL2\data\neg_test"
OUTPUT_DIR       = "stratified_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_split(pos_path, neg_path):
    pos = np.load(pos_path).astype(np.float32)
    neg = np.load(neg_path).astype(np.float32)
    x   = np.concatenate([pos, neg], axis=0)
    y   = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(int)
    x, y = shuffle(x, y, random_state=42)
    return x, tf.keras.utils.to_categorical(y, NUM_CLASSES), len(pos), len(neg)


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end   = min(start + self.batch_size, len(self.data))
        return self.data[start:end], self.labels[start:end]


class DeepScan(Model):
    def __init__(self):
        super(DeepScan, self).__init__()
        self.conv_layers = []
        self.pool_layers = []
        self.flat_layers = []

        for ws in [8, 16]:
            self.conv_layers.append(layers.Conv2D(
                NUM_FILTER, kernel_size=(1, ws), activation='relu', padding='valid',
                bias_initializer=tf.constant_initializer(0.1),
                kernel_initializer='glorot_uniform'))
            self.pool_layers.append(layers.MaxPooling2D(
                pool_size=(1, MAXSEQ - ws + 1), strides=(1, MAXSEQ), padding='valid'))
            self.flat_layers.append(layers.Flatten())

        self.concat  = layers.Concatenate(axis=1)
        self.dropout = layers.Dropout(0.7)
        self.fc1 = layers.Dense(NUM_HIDDEN, activation='relu',
                                bias_initializer=tf.constant_initializer(0.1),
                                kernel_initializer='glorot_uniform')
        self.fc2 = layers.Dense(NUM_CLASSES, activation='softmax',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, x, training=False):
        branches = [self.flat_layers[i](self.pool_layers[i](self.conv_layers[i](x)))
                    for i in range(2)]
        x = self.concat(branches)
        x = self.dropout(x, training=training)
        return self.fc2(self.fc1(x))


def get_threshold(model, pos_emb, neg_emb):
    x_all = np.concatenate([pos_emb, neg_emb], axis=0)
    y_all = np.concatenate([np.ones(len(pos_emb)), np.zeros(len(neg_emb))])
    pred  = model.predict(x_all, verbose=0)

    fpr, tpr, thresholds = roc_curve(y_all, pred[:, 1])
    auc       = metrics.auc(fpr, tpr)
    threshold = thresholds[np.argmax(np.sqrt(tpr * (1 - fpr)))]
    print(f"  Threshold: {threshold:.4f}  AUC: {auc:.4f}")
    return threshold, auc


def specificity(model, neg_emb, threshold):
    pred   = model.predict(neg_emb, verbose=0)
    y_pred = (pred[:, 1] >= threshold).astype(int)
    TN = int((y_pred == 0).sum())
    FP = int((y_pred == 1).sum())
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return spec, TN, FP, TN + FP


def main():
    print("Loading training data...")
    x_train, y_train, n_pos, n_neg = load_split(POS_TRAIN, NEG_TRAIN)
    print(f"  pos={n_pos}  neg={n_neg}  total={len(x_train)}")

    print("\nTraining model (seed=42)...")
    tf.keras.backend.clear_session()
    gc.collect()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    gen = DataGenerator(x_train, y_train, BATCH_SIZE)
    model = DeepScan()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.build(input_shape=x_train.shape)
    model.fit(gen, epochs=EPOCHS, shuffle=True, verbose=1)

    print("\nLoading test embeddings...")
    pos_test = np.load(POS_TEST).astype(np.float32)
    neg_test = np.load(NEG_TEST).astype(np.float32)
    print(f"  pos_test={pos_test.shape}  neg_test={neg_test.shape}")

    print("\nFinding optimal threshold...")
    threshold, auc = get_threshold(model, pos_test, neg_test)

    spec_all, TN_all, FP_all, N_all = specificity(model, neg_test, threshold)
    print(f"  Overall specificity: {spec_all:.4f}  TN={TN_all}  FP={FP_all}  N={N_all}")

    print("\nAssigning negative source labels...")
    fasta_files = sorted([f for f in os.listdir(NEG_TEST_DIR) if f.endswith(".fasta")])
    ordered_ids = [os.path.splitext(f)[0] for f in fasta_files]

    il2_ids      = set(pd.read_csv(NEG_IL2_CSV)["Epitope_ID"].astype(str).str.strip())
    cytokine_ids = set(pd.read_csv(NEG_CYTOKINE_CSV)["Epitope_ID"].astype(str).str.strip())

    il2_idx, cyt_idx = [], []
    for i, seq_id in enumerate(ordered_ids):
        if seq_id in il2_ids:
            il2_idx.append(i)
        elif seq_id in cytokine_ids:
            cyt_idx.append(i)

    print(f"  IL-2 inactive: {len(il2_idx)}  Other cytokines: {len(cyt_idx)}")

    spec_il2, TN_il2, FP_il2, N_il2 = specificity(model, neg_test[il2_idx], threshold)
    spec_cyt, TN_cyt, FP_cyt, N_cyt = specificity(model, neg_test[cyt_idx], threshold)

    print(f"\n  IL-2 inactive:    Spec={spec_il2:.4f}  TN={TN_il2}  FP={FP_il2}  N={N_il2}")
    print(f"  Other cytokines:  Spec={spec_cyt:.4f}  TN={TN_cyt}  FP={FP_cyt}  N={N_cyt}")

    out_csv = os.path.join(OUTPUT_DIR, "stratified_specificity.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["negative_source", "specificity", "TN", "FP", "total",
                         "threshold", "AUC"])
        writer.writerow(["Overall",
                         round(spec_all, 4), TN_all, FP_all, N_all,
                         round(threshold, 4), round(auc, 4)])
        writer.writerow(["IL-2 inactive variants",
                         round(spec_il2, 4), TN_il2, FP_il2, N_il2,
                         round(threshold, 4), round(auc, 4)])
        writer.writerow(["Other cytokines (IL-4/6/7/10/21)",
                         round(spec_cyt, 4), TN_cyt, FP_cyt, N_cyt,
                         round(threshold, 4), round(auc, 4)])

    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
