"""
Model Comparison Script
-----------------------
This script loads the trained models and preprocessed datasets,
evaluates them on train/validation/test sets, and shows a
comparative performance table.

Models compared:
- KNN
- Random Forest
- Gradient Boosting
- Logistic Regression
- Deep Neural Network (DNN)
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

# ==========================
# Load processed datasets
# ==========================
X_train = joblib.load("X_train.pkl")
X_val = joblib.load("X_val.pkl")
X_test = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_val = joblib.load("y_val.pkl")
y_test = joblib.load("y_test.pkl")

# ==========================
# Remap labels for DNN
# ==========================
unique_labels = sorted(set(y_train) | set(y_val) | set(y_test))
label_map = {label: idx for idx, label in enumerate(unique_labels)}

y_train_dnn = np.array([label_map[y] for y in y_train])
y_val_dnn = np.array([label_map[y] for y in y_val])
y_test_dnn = np.array([label_map[y] for y in y_test])

# ==========================
# Load trained models
# ==========================
models = {
    "KNN": joblib.load("knn_model.pkl"),
    "Random Forest": joblib.load("random_forest_tuned.pkl"),
    "Gradient Boosting": joblib.load("gb_model.pkl"),
    "Logistic Regression": joblib.load("logreg_best.pkl"),
    "DNN": load_model("dnn_model.h5")
}

# ==========================
# Evaluate models
# ==========================
results = []

for name, model in models.items():
    if name == "DNN":
        # Predictions with DNN
        y_train_pred = model.predict(X_train).argmax(axis=1)
        y_val_pred = model.predict(X_val).argmax(axis=1)
        y_test_pred = model.predict(X_test).argmax(axis=1)

        results.append({
            "Model": name,
            "Train Accuracy": accuracy_score(y_train_dnn, y_train_pred),
            "Validation Accuracy": accuracy_score(y_val_dnn, y_val_pred),
            "Test Accuracy": accuracy_score(y_test_dnn, y_test_pred)
        })
    else:
        # Predictions with sklearn models
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Train Accuracy": accuracy_score(y_train, y_train_pred),
            "Validation Accuracy": accuracy_score(y_val, y_val_pred),
            "Test Accuracy": accuracy_score(y_test, y_test_pred)
        })

# ==========================
# Display results
# ==========================
df_results = pd.DataFrame(results)
print("\nComparative Results:\n")
print(df_results)

# ==========================
# Save results to CSV
# ==========================
df_results.to_csv("comparative_results.csv", index=False)
print("\nResults saved to comparative_results.csv")