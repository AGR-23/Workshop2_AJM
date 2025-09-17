"""
Deep Neural Network (DNN) Training Script
-----------------------------------------
This script trains a deep neural network on the preprocessed
student academic stress dataset.

Requirements:
- preprocessing.py (with SMOTE balancing)
- TensorFlow/Keras

Steps:
1. Load preprocessed train/val/test sets
2. Adjust labels to start from 0 (for categorical crossentropy)
3. Build and compile DNN model (3+ hidden layers, regularization)
4. Train with early stopping
5. Evaluate on train/val/test
6. Save model
"""

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# ==========================
# Load processed data
# ==========================
X_train = joblib.load("X_train.pkl")
X_val = joblib.load("X_val.pkl")
X_test = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_val = joblib.load("y_val.pkl")
y_test = joblib.load("y_test.pkl")

# ==========================
# Fix labels (shift to 0-based)
# ==========================
y_train = np.array(y_train) - 1
y_val = np.array(y_val) - 1
y_test = np.array(y_test) - 1

num_classes = len(np.unique(y_train))
input_dim = X_train.shape[1]

print(f"Input dim: {input_dim}, Classes: {num_classes}")

# ==========================
# Build DNN model
# ==========================
model = Sequential([
    Input(shape=(input_dim,)),
    Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================
# Training
# ==========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# ==========================
# Evaluation
# ==========================
train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
test_acc = model.evaluate(X_test, y_test, verbose=0)[1]

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ==========================
# Save model
# ==========================
model.save("dnn_model.h5")
print("DNN model saved as dnn_model.h5")