import os
import gc
import logging
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_split(pos_path, neg_path):
    # load positive and negative embeddings then stack them
    pos = np.load(pos_path)
    neg = np.load(neg_path)
    x = np.concatenate([pos, neg], axis=0)
    # make labels: 1 for positive, 0 for negative
    labels = np.concatenate([np.ones(pos.shape[0]),
                             np.zeros(neg.shape[0])], axis=0)
    y = tf.keras.utils.to_categorical(labels, 2)
    gc.collect()
    return x, y


def MCNN_data_load(base_path):
    pos_train = os.path.join(base_path, "pos_real_train_rag.npy")
    pos_test  = os.path.join(base_path, "pos_real_test_rag.npy")
    neg_train = os.path.join(base_path, "neg_real_train_rag.npy")
    neg_test  = os.path.join(base_path, "neg_real_test_rag.npy")

    for p in [pos_train, pos_test, neg_train, neg_test]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Data file not found: {p}")

    logging.info(f"Loading data from {base_path}")
    x_train, y_train = load_split(pos_train, neg_train)
    x_train, y_train = shuffle(x_train, y_train, random_state=42)

    x_test, y_test = load_split(pos_test, neg_test)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)

    logging.info(f"Train: {x_train.shape}  Test: {x_test.shape}")
    return x_train, y_train, x_test, y_test
