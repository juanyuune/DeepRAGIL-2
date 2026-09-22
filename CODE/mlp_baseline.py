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

# configuration
NUM_FEATURE = 1280
NUM_HIDDEN  = 500
BATCH_SIZE  = 512
NUM_CLASSES = 2
EPOCHS      = 20
SEEDS       = list(range(1, 11))

# raw ESM-2 embeddings, no RAG
BASE_DATA  = r"C:\jupyter\juan\IL2\dataset\esm2\max_35"
OUTPUT_DIR = "mlp_baseline_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(base_path):
    pos_train = np.load(os.path.join(base_path, "pos_real_train.npy"))
    neg_train = np.load(os.path.join(base_path, "neg_real_train.npy"))
    pos_test  = np.load(os.path.join(base_path, "pos_real_test.npy"))
    neg_test  = np.load(os.path.join(base_path, "neg_real_test.npy"))

    x_train = np.concatenate([pos_train, neg_train], axis=0).astype(np.float32)
    x_test  = np.concatenate([pos_test,  neg_test],  axis=0).astype(np.float32)

    y_train_raw = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))])
    y_test_raw  = np.concatenate([np.ones(len(pos_test)),  np.zeros(len(neg_test))])

    x_train, y_train_raw = shuffle(x_train, y_train_raw, random_state=42)
    x_test,  y_test_raw  = shuffle(x_test,  y_test_raw,  random_state=42)

    # mean pool: (N, 1, 35, 1280) -> (N, 1280)
    x_train = x_train.squeeze(axis=1).mean(axis=1)
    x_test  = x_test.squeeze(axis=1).mean(axis=1)

    y_train = tf.keras.utils.to_categorical(y_train_raw, NUM_CLASSES)
    y_test  = tf.keras.utils.to_categorical(y_test_raw,  NUM_CLASSES)

    print(f"Train: {x_train.shape}  Test: {x_test.shape}")
    return x_train, y_train, x_test, y_test


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


class MLPBaseline(Model):
    def __init__(self):
        super(MLPBaseline, self).__init__()
        self.fc1     = layers.Dense(NUM_HIDDEN, activation='relu',
                                    bias_initializer=tf.constant_initializer(0.1),
                                    kernel_initializer='glorot_uniform')
        self.dropout = layers.Dropout(0.7)
        self.fc2     = layers.Dense(NUM_CLASSES, activation='softmax',
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        return self.fc2(x)


def get_metrics(model, x_test, y_test):
    pred = model.predict(x_test, verbose=0)

    fpr, tpr, thresholds = roc_curve(y_test[:, 1], pred[:, 1])
    auc = metrics.auc(fpr, tpr)

    best_idx  = np.argmax(np.sqrt(tpr * (1 - fpr)))
    threshold = thresholds[best_idx]
    y_pred    = (pred[:, 1] >= threshold).astype(int)

    TN, FP, FN, TP = metrics.confusion_matrix(y_test[:, 1], y_pred).ravel()
    Sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    Spec = TN / (FP + TN) if (FP + TN) > 0 else 0.0
    Acc  = (TP + TN) / (TP + FP + TN + FN)

    denom = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    MCC   = (TP*TN - FP*FN) / denom if denom > 0 else 0.0

    return Sens, Spec, Acc, MCC, auc, int(TP), int(FP), int(TN), int(FN)


def main():
    print("Loading raw ESM-2 data (no RAG)...")
    x_train, y_train, x_test, y_test = load_data(BASE_DATA)

    all_results  = []
    per_seed_csv = os.path.join(OUTPUT_DIR, "mlp_per_seed_results.csv")

    with open(per_seed_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "seed", "Sens", "Spec", "Acc", "MCC", "AUC",
            "TP", "FP", "TN", "FN", "duration"
        ])

    for seed in SEEDS:
        print(f"\nSeed {seed}/{len(SEEDS)}")
        t_start = datetime.datetime.now()

        tf.keras.backend.clear_session()
        gc.collect()
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        gen = DataGenerator(x_train, y_train, BATCH_SIZE)

        model = MLPBaseline()
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.build(input_shape=(None, NUM_FEATURE))
        model.fit(gen, epochs=EPOCHS, shuffle=True, verbose=1)

        result = get_metrics(model, x_test, y_test)
        duration = datetime.datetime.now() - t_start

        Sens, Spec, Acc, MCC, AUC, TP, FP, TN, FN = result
        print(f"  Sens={Sens:.4f}  Spec={Spec:.4f}  Acc={Acc:.4f}  "
              f"MCC={MCC:.4f}  AUC={AUC:.4f}")
        print(f"  TP={TP}  FP={FP}  TN={TN}  FN={FN}")
        print(f"  Duration: {duration}")

        all_results.append([Sens, Spec, Acc, MCC, AUC])

        with open(per_seed_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                seed,
                round(Sens, 4), round(Spec, 4), round(Acc, 4),
                round(MCC,  4), round(AUC,  4),
                TP, FP, TN, FN, str(duration)
            ])

        model.save_weights(os.path.join(OUTPUT_DIR, f"mlp_weights_seed{seed}.weights.h5"))
        del model
        gc.collect()

    arr  = np.array(all_results)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)

    names = ["Sens", "Spec", "Acc", "MCC", "AUC"]
    print(f"\nMLP summary across {len(SEEDS)} seeds:")
    for i, name in enumerate(names):
        print(f"  {name}: {mean[i]:.4f} +/- {std[i]:.4f}")

    summary_csv = os.path.join(OUTPUT_DIR, "mlp_summary_mean_std.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        for i, name in enumerate(names):
            writer.writerow([name, round(mean[i], 4), round(std[i], 4)])

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
