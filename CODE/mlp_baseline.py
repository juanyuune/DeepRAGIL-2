import os
import gc
import csv
import math
import random
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.utils import shuffle

# ── configuration — identical to DeepRAGIL-2 except architecture ──
NUM_FEATURE = 1280
NUM_HIDDEN  = 500
BATCH_SIZE  = 512
NUM_CLASSES = 2
EPOCHS      = 20

# ── paths — raw ESM-2 embeddings, NO RAG fusion ──
BASE_DATA  = r"C:\jupyter\juan\IL2\dataset\esm2\max_35"
OUTPUT_DIR = "mlp_baseline_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# ── data loader ──
def load_data(base_path):
    pos_train = np.load(os.path.join(base_path, "pos_real_train.npy"))
    neg_train = np.load(os.path.join(base_path, "neg_real_train.npy"))
    pos_test  = np.load(os.path.join(base_path, "pos_real_test.npy"))
    neg_test  = np.load(os.path.join(base_path, "neg_real_test.npy"))

    x_train = np.concatenate([pos_train, neg_train], axis=0)
    x_test  = np.concatenate([pos_test,  neg_test],  axis=0)

    y_train_raw = np.concatenate([
        np.ones(pos_train.shape[0]),
        np.zeros(neg_train.shape[0])
    ])
    y_test_raw = np.concatenate([
        np.ones(pos_test.shape[0]),
        np.zeros(neg_test.shape[0])
    ])

    x_train, y_train_raw = shuffle(x_train, y_train_raw, random_state=42)
    x_test,  y_test_raw  = shuffle(x_test,  y_test_raw,  random_state=42)

    # mean-pool across sequence positions: (N, 1, 35, 1280) -> (N, 1280)
    # step 1: squeeze channel dim (axis=1) -> (N, 35, 1280)
    # step 2: mean across sequence length (axis=1) -> (N, 1280)
    x_train = x_train.squeeze(axis=1).mean(axis=1)
    x_test  = x_test.squeeze(axis=1).mean(axis=1)
    assert x_train.shape[1] == NUM_FEATURE, \
        f"Expected feature dim {NUM_FEATURE}, got {x_train.shape[1]}"
    assert x_test.shape[1] == NUM_FEATURE, \
        f"Expected feature dim {NUM_FEATURE}, got {x_test.shape[1]}" 

    y_train = tf.keras.utils.to_categorical(y_train_raw, NUM_CLASSES)
    y_test  = tf.keras.utils.to_categorical(y_test_raw,  NUM_CLASSES)

    print(f"Train: {x_train.shape}  Test: {x_test.shape}")
    return x_train, y_train, x_test, y_test


# ── data generator ──
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, labels, batch_size):
        self.data       = data
        self.labels     = labels
        self.batch_size = batch_size
        self.indexes    = np.arange(len(self.data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        idx          = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data   = np.array([self.data[i]   for i in idx])
        batch_labels = np.array([self.labels[i] for i in idx])
        return batch_data, batch_labels


# ── MLP model ──
class MLPBaseline(Model):
    def __init__(self, num_feature=NUM_FEATURE,
                 num_hidden=NUM_HIDDEN,
                 num_classes=NUM_CLASSES):
        super(MLPBaseline, self).__init__()

        self.fc1 = layers.Dense(
            num_hidden,
            activation='relu',
            bias_initializer=tf.constant_initializer(0.1),
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )
        self.dropout = layers.Dropout(rate=0.7)
        self.fc2 = layers.Dense(
            num_classes,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(1e-3)
        )

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return x


# ── evaluation — identical to multiseed_experiment.py ──
def evaluate(model, x_test, y_test):
    pred = model.predict(x_test, verbose=0)

    fpr, tpr, thresholds = roc_curve(y_test[:, 1], pred[:, 1])
    auc = metrics.auc(fpr, tpr)

    gmeans    = np.sqrt(tpr * (1 - fpr))
    ix        = np.argmax(gmeans)
    threshold = thresholds[ix]

    y_pred = (pred[:, 1] >= threshold).astype(int)
    TN, FP, FN, TP = metrics.confusion_matrix(y_test[:, 1], y_pred).ravel()

    Sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    Spec = TN / (FP + TN) if (FP + TN) > 0 else 0.0
    Acc  = (TP + TN) / (TP + FP + TN + FN)
    MCC  = (TP * TN - FP * FN) / math.sqrt(
               (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
           ) if (TP + FP) > 0 and (FP + TN) > 0 and (TP + FN) > 0 and (TN + FN) > 0 else 0.0

    return Sens, Spec, Acc, MCC, auc, TP, FP, TN, FN


# ── main ──
def main():
    print("Loading data (raw ESM-2, no RAG)...")
    x_train, y_train, x_test, y_test = load_data(BASE_DATA)

    all_results  = []
    per_seed_csv = os.path.join(OUTPUT_DIR, "mlp_per_seed_results.csv")

    with open(per_seed_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "Sens", "Spec", "Acc", "MCC", "AUC",
            "TP", "FP", "TN", "FN", "duration"
        ])

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  MLP Seed {seed}/{len(SEEDS)}")
        print(f"{'='*60}")

        # clear session first, then set all seeds
        tf.keras.backend.clear_session()
        gc.collect()

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        t_start = datetime.datetime.now()

        generator = DataGenerator(x_train, y_train, batch_size=BATCH_SIZE)

        model = MLPBaseline(
            num_feature=NUM_FEATURE,
            num_hidden=NUM_HIDDEN,
            num_classes=NUM_CLASSES
        )
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.build(input_shape=(None, NUM_FEATURE))

        model.fit(
            generator,
            epochs=EPOCHS,
            shuffle=True,
            verbose=1
        )

        Sens, Spec, Acc, MCC, AUC, TP, FP, TN, FN = evaluate(model, x_test, y_test)
        duration = datetime.datetime.now() - t_start

        print(f"\nMLP Seed {seed} results:")
        print(f"  Sens={Sens:.4f}  Spec={Spec:.4f}  Acc={Acc:.4f}  MCC={MCC:.4f}  AUC={AUC:.4f}")
        print(f"  TP={TP}  FP={FP}  TN={TN}  FN={FN}")
        print(f"  Duration: {duration}")

        all_results.append([Sens, Spec, Acc, MCC, AUC])

        with open(per_seed_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                seed,
                round(Sens, 4), round(Spec, 4), round(Acc, 4),
                round(MCC,  4), round(AUC,  4),
                TP, FP, TN, FN,
                str(duration)
            ])

        weights_path = os.path.join(
            OUTPUT_DIR, f"mlp_weights_seed{seed}.weights.h5"
        )
        model.save_weights(weights_path)

        del model
        gc.collect()

    # ── summary ──
    arr  = np.array(all_results)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)

    print(f"\n{'='*60}")
    print(f"  MLP SUMMARY ACROSS {len(SEEDS)} SEEDS")
    print(f"{'='*60}")
    metrics_names = ["Sens", "Spec", "Acc", "MCC", "AUC"]
    for i, name in enumerate(metrics_names):
        print(f"  {name}: {mean[i]:.4f} ± {std[i]:.4f}")

    summary_csv = os.path.join(OUTPUT_DIR, "mlp_summary_mean_std.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        for i, name in enumerate(metrics_names):
            writer.writerow([name, round(mean[i], 4), round(std[i], 4)])

    # ── print final comparison table ──
    print(f"\n{'='*60}")
    print(f"  FINAL TABLE FOR MANUSCRIPT")
    print(f"{'='*60}")
    print(f"  {'Model':<30} {'Sens':>6} {'Spec':>6} {'Acc':>6} {'MCC':>6} {'AUC':>6}")
    print(f"  {'-'*60}")
    print(f"  {'MLP (ESM-2, no RAG/MCNN)':<30} "
          f"{mean[0]:>6.4f} {mean[1]:>6.4f} {mean[2]:>6.4f} "
          f"{mean[3]:>6.4f} {mean[4]:>6.4f}")
    print(f"  {'  (± SD)':<30} "
          f"{std[0]:>6.4f} {std[1]:>6.4f} {std[2]:>6.4f} "
          f"{std[3]:>6.4f} {std[4]:>6.4f}")

    print(f"\nPer-seed results : {per_seed_csv}")
    print(f"Summary (mean±SD): {summary_csv}")


if __name__ == "__main__":
    main()
