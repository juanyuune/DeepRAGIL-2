import os
import gc
import csv
import math
import datetime
import pickle
import time
from time import gmtime, strftime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold

import import_data_esm2_old as load_data


# --- params ---
DATA_LABEL     = load_data.data_label()
DATA_TYPE      = "esm2"
MAXSEQ         = 35
NUM_FEATURE    = 1280
NUM_FILTER     = 256
NUM_HIDDEN     = 500
BATCH_SIZE     = 512
WINDOW_SIZES   = [8, 16]
NUM_CLASSES    = 2
CLASS_NAMES    = ['Negative', 'Positive']
EPOCHS         = 20
K_FOLD         = 5
VALIDATION_MODE = "independent"
# options: "independent", "cross"
IMBALANCE      = "RANDOM"
# options: None, "SMOTE", "ADASYN", "RANDOM"


# --- time log ---
write_data = []
start_time = datetime.datetime.now()
write_data.append(time.ctime())
write_data.append(DATA_LABEL)
write_data.append(DATA_TYPE)
write_data.append(BATCH_SIZE)
write_data.append(NUM_HIDDEN)
write_data.append(WINDOW_SIZES)
write_data.append(NUM_FILTER)
write_data.append(VALIDATION_MODE)
write_data.append(IMBALANCE)


def time_log(message):
    print(message, " : ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))


# --- PKL save ---
def save_roc(fpr, tpr, auc):
    folder = "./PKL/rag/"
    os.makedirs(folder, exist_ok=True)
    fname = f"0.5_RANDOM_ESM2_RAG_MCNN_Independent_8,16_{int(time.time())}.pkl"
    fpath = os.path.join(folder, fname)
    with open(fpath, "wb") as f:
        pickle.dump({"fpr": fpr, "tpr": tpr, "AUC": auc}, f)
    print(f"ROC data saved to: {os.path.abspath(fpath)}")


# --- data generator ---
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, labels, batch_size):
        self.data       = data
        self.labels     = labels
        self.batch_size = batch_size
        self.indexes    = np.arange(len(self.data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        idx         = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data  = np.array([self.data[i]   for i in idx])
        batch_labels = np.array([self.labels[i] for i in idx])
        return batch_data, batch_labels


# --- model ---
class DeepScan(Model):

    def __init__(self, input_shape=(1, MAXSEQ, NUM_FEATURE),
                 window_sizes=[32], num_filters=256, num_hidden=1000):
        super(DeepScan, self).__init__()
        self.input_layer  = tf.keras.Input(input_shape)
        self.window_sizes = window_sizes
        self.conv2d   = []
        self.maxpool  = []
        self.flatten  = []

        for ws in self.window_sizes:
            self.conv2d.append(layers.Conv2D(
                filters=num_filters,
                kernel_size=(1, ws),
                activation=tf.nn.relu,
                padding='valid',
                bias_initializer=tf.constant_initializer(0.1),
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ))
            self.maxpool.append(layers.MaxPooling2D(
                pool_size=(1, MAXSEQ - ws + 1),
                strides=(1, MAXSEQ),
                padding='valid'
            ))
            self.flatten.append(layers.Flatten())

        self.dropout = layers.Dropout(rate=0.7)
        self.fc1 = layers.Dense(
            num_hidden,
            activation=tf.nn.relu,
            bias_initializer=tf.constant_initializer(0.1),
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )
        self.fc2 = layers.Dense(
            NUM_CLASSES,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(1e-3)
        )
        self.out = self.call(self.input_layer)

    def call(self, x, training=False):
        _x = []
        for i in range(len(self.window_sizes)):
            x_conv = self.conv2d[i](x)
            x_maxp = self.maxpool[i](x_conv)
            x_flat = self.flatten[i](x_maxp)
            _x.append(x_flat)

        x = tf.concat(_x, 1)
        x = self.dropout(x, training=training)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# --- imbalance handler ---
def handle_imbalance(mode, x_train, y_train):
    if mode == "None" or mode is None:
        return x_train, y_train

    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

    x_2d = x_train.reshape(x_train.shape[0], -1)
    print(x_2d.shape)
    print(y_train.shape)

    if mode == "SMOTE":
        sampler = SMOTE(random_state=42)
    elif mode == "ADASYN":
        sampler = ADASYN(random_state=42)
    else:
        sampler = RandomOverSampler(random_state=42)

    x_res, y_res = sampler.fit_resample(x_2d, y_train)
    x_res = x_res.reshape(x_res.shape[0], 1, MAXSEQ, NUM_FEATURE)

    print(x_res.shape)
    print(y_res.shape)

    del x_2d
    gc.collect()

    y_res = tf.keras.utils.to_categorical(y_res, NUM_CLASSES)
    return x_res, y_res


# --- evaluation ---
def model_test(model, x_test, y_test):
    print(x_test.shape)
    pred = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test[:, 1], pred[:, 1])
    auc = metrics.auc(fpr, tpr)

    disp = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name='mCNN')
    disp.plot()

    gmeans = np.sqrt(tpr * (1 - fpr))
    ix     = np.argmax(gmeans)
    print(f'Best Threshold={thresholds[ix]}, G-Mean={gmeans[ix]}')
    threshold = thresholds[ix]

    y_pred = (pred[:, 1] >= threshold).astype(int)
    TN, FP, FN, TP = metrics.confusion_matrix(y_test[0:][:, 1], y_pred).ravel()

    Sens = TP / (TP + FN) if TP + FN > 0 else 0.0
    Spec = TN / (FP + TN) if FP + TN > 0 else 0.0
    Acc  = (TP + TN) / (TP + FP + TN + FN)
    MCC  = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) \
           if TP + FP > 0 and FP + TN > 0 and TP + FN and TN + FN else 0.0
    F1   = 2 * TP / (2 * TP + FP + FN)
    Prec   = TP / (TP + FP)
    Recall = TP / (TP + FN)

    print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, '
          f'Acc={Acc:.4f}, MCC={MCC:.4f}, AUC={auc:.4f}, '
          f'F1={F1:.4f}, Prec={Prec:.4f}, Recall={Recall:.4f}\n')

    save_roc(fpr, tpr, auc)
    return TP, FP, TN, FN, Sens, Spec, Acc, MCC, auc


# --- load data ---
x_train, y_train, x_test, y_test = load_data.MCNN_data_load(DATA_TYPE, MAXSEQ)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# --- training ---
if VALIDATION_MODE == "cross":
    time_log("Start cross")
    kfold   = KFold(n_splits=K_FOLD, shuffle=True, random_state=2)
    results = []
    i = 1
    for train_idx, test_idx in kfold.split(x_train):
        print(i, "/", K_FOLD, '\n')
        X_train, X_test = x_train[train_idx], x_train[test_idx]
        Y_train, Y_test = y_train[train_idx], y_train[test_idx]
        print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

        X_train, Y_train = handle_imbalance(IMBALANCE, X_train, Y_train)
        generator = DataGenerator(X_train, Y_train, batch_size=BATCH_SIZE)

        model = DeepScan(num_filters=NUM_FILTER, num_hidden=NUM_HIDDEN,
                         window_sizes=WINDOW_SIZES)
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.build(input_shape=X_train.shape)
        model.fit(
            generator,
            epochs=EPOCHS,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)],
            verbose=1,
            shuffle=True
        )

        TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC = model_test(model, X_test, Y_test)
        results.append([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC])
        i += 1

        del X_train, X_test, Y_train, Y_test
        gc.collect()

    mean_r = np.mean(results, axis=0)
    print(f'TP={mean_r[0]:.4}, FP={mean_r[1]:.4}, TN={mean_r[2]:.4}, FN={mean_r[3]:.4}, '
          f'Sens={mean_r[4]:.4}, Spec={mean_r[5]:.4}, Acc={mean_r[6]:.4}, '
          f'MCC={mean_r[7]:.4}, AUC={mean_r[8]:.4}\n')
    write_data.extend(mean_r)


if VALIDATION_MODE == "independent":
    x_train, y_train = handle_imbalance(IMBALANCE, x_train, y_train)
    generator = DataGenerator(x_train, y_train, batch_size=BATCH_SIZE)

    time_log("Start Model Train")
    model = DeepScan(num_filters=NUM_FILTER, num_hidden=NUM_HIDDEN,
                     window_sizes=WINDOW_SIZES)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.build(input_shape=x_train.shape)
    model.summary()
    model.fit(generator, epochs=EPOCHS, shuffle=True)
    time_log("End Model Train")

    time_log("Start Model Test")
    TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC = model_test(model, x_test, y_test)
    write_data.extend([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC])
    time_log("End Model Test")


if VALIDATION_MODE == "LOAD":
    model = DeepScan(num_filters=NUM_FILTER, num_hidden=NUM_HIDDEN,
                     window_sizes=WINDOW_SIZES)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.build(input_shape=x_train.shape)
    model.summary()
    model.load_weights('my_model_weights.h5')

    time_log("Start Model Test")
    TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC = model_test(model, x_test, y_test)
    write_data.extend([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC])
    time_log("End Model Test")


# --- save results to CSV ---
def save_csv(write_data, start):
    end = datetime.datetime.now()
    write_data.append(end - start)
    fpath = "./results/MAX35_PLM_RAG_MCNN.csv"
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "a", newline="") as f:
        csv.writer(f).writerow(write_data)

save_csv(write_data, start_time)


# --- save model weights ---
os.makedirs("./saved_weights/model/rag", exist_ok=True)
weights_path = f"./saved_weights/model/rag/0.5_RANDOM_{MAXSEQ}_{DATA_TYPE}_{WINDOW_SIZES}.h5"
model.save_weights(weights_path)
print(f"model saved: {weights_path}")
