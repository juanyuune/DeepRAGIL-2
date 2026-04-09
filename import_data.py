import os
import gc
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

datalabel = "IL2"


def MCNN_data_load(DATA_TYPE, MAXSEQ,
                   base_path="dataset/esm2/max_35/0.5_rag_fused"):

    pos_train = os.path.join(base_path, "pos_real_train_rag.npy")
    pos_test  = os.path.join(base_path, "pos_real_test_rag.npy")
    neg_train = os.path.join(base_path, "neg_real_train_rag.npy")
    neg_test  = os.path.join(base_path, "neg_real_test_rag.npy")

    x_train, y_train = load_split(pos_train, neg_train)
    x_train, y_train = shuffle(x_train, y_train, random_state=42)

    x_test, y_test = load_split(pos_test, neg_test)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)

    return x_train, y_train, x_test, y_test


def load_split(pos_path, neg_path):
    # load positive and negative embeddings then stack them
    pos = np.load(pos_path)
    neg = np.load(neg_path)

    x = np.concatenate([pos, neg], axis=0)

    # make labels: 1 for positive, 0 for negative
    labels = np.concatenate([np.ones(pos.shape[0]),
                             np.zeros(neg.shape[0])], axis=0)
    y = tf.keras.utils.to_categorical(labels, 2)

    gc.collect()  # free memory before returning large arrays
    return x, y