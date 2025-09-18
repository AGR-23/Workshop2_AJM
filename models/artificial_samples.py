"""
Point 5 – Artificial Sample Test
--------------------------------
This script implements requirement 5 of the lab: generate an invented (artificial)
sample, feed it to the *selected/best* model, and analyze the prediction — also
testing counterfactual changes on key directions of variation.

What it does:
1) Selects the best model from `comparative_results.csv` (highest Test Accuracy,
   then Validation Accuracy). If TensorFlow is unavailable, it will skip the DNN.
2) Loads the preprocessed, encoded, and scaled features produced in preprocessing.
3) Creates one artificial point close to the global mean of the (encoded/scaled)
   feature space and predicts its class + probabilities.
4) Perturbs that point along the top-3 principal directions (±) to simulate
   "what-if" changes, storing predictions for each scenario.
5) Saves a CSV (`point5_artificial_results.csv`) with the scenario name,
   predicted class, and per-class probabilities. Also saves `point5_vectors.npy`
   with the underlying vectors for reproducibility.

Inputs required (already produced in earlier steps of the project):
- X_train.pkl, X_val.pkl, X_test.pkl, y_train.pkl, y_val.pkl, y_test.pkl
- comparative_results.csv
- Trained models (any subset): knn_model.pkl, random_forest_final_model.pkl,
  gb_model.pkl, logreg_best.pkl, dnn_model.h5

"""

import os
import joblib
import numpy as np
import pandas as pd

# ---------------------------
# Optional TensorFlow check
# ---------------------------
def tf_available() -> bool:
    try:
        import tensorflow as tf  # noqa: F401
        return True
    except Exception:
        return False

# ---------------------------
# Model loading utilities
# ---------------------------
def load_model_auto(name: str, base: str = "."):
    """
    Load a trained model by its name. Supports scikit-learn and (optionally) Keras.
    Returns (model, kind) where kind is "sk" or "tf".
    """
    path_map = {
        "KNN": ("knn_model.pkl", "sk"),
        "Random Forest": ("random_forest_final_model.pkl", "sk"),
        "Gradient Boosting": ("gb_model.pkl", "sk"),
        "Logistic Regression": ("logreg_best.pkl", "sk"),
        "DNN": ("dnn_model.h5", "tf"),
    }
    if name not in path_map:
        raise ValueError(f"Unrecognized model name: {name}")
    fname, kind = path_map[name]
    fpath = os.path.join(base, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Model file not found: {fpath}")
    if kind == "sk":
        return joblib.load(fpath), kind
    else:
        if not tf_available():
            raise RuntimeError("TensorFlow/Keras is not available to load the DNN.")
        from tensorflow.keras.models import load_model as _load_model
        return _load_model(fpath), kind

def predict_proba_any(model, kind: str, X: np.ndarray) -> np.ndarray:
    """
    Return class probabilities for both sklearn and tf models.
    Falls back to decision_function or one-hot argmax if needed.
    """
    if kind == "sk":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        elif hasattr(model, "decision_function"):
            z = model.decision_function(X)
            e = np.exp(z - z.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        else:
            preds = model.predict(X)
            classes = np.unique(preds)
            proba = np.zeros((X.shape[0], len(classes)))
            for i, p in enumerate(preds):
                proba[i, np.where(classes == p)[0][0]] = 1.0
            return proba
    else:
        return model.predict(X, verbose=0)

# ---------------------------
# Select the best model
# ---------------------------
BASE = "."
comp_path = os.path.join(BASE, "comparative_results.csv")
if not os.path.exists(comp_path):
    raise FileNotFoundError("comparative_results.csv not found. Run model comparison first.")

comp = pd.read_csv(comp_path)
comp.columns = [c.strip() for c in comp.columns]

# Order by Test Accuracy (desc), then Validation Accuracy (desc)
comp_sorted = comp.sort_values(["Test Accuracy", "Validation Accuracy"],
                               ascending=[False, False]).reset_index(drop=True)

candidates = comp_sorted["Model"].tolist()
if not tf_available():
    # If TF is missing, prefer sklearn models first
    candidates = [m for m in candidates if m != "DNN"] + [m for m in candidates if m == "DNN"]

selected_model_name = None
model = None
kind = None
for m in candidates:
    try:
        model, kind = load_model_auto(m, BASE)
        selected_model_name = m
        break
    except Exception:
        continue

if model is None:
    raise RuntimeError("Could not load any model listed in comparative_results.csv.")

# ---------------------------
# Load preprocessed feature sets
# ---------------------------
X_train = joblib.load(os.path.join(BASE, "X_train.pkl"))
y_train = joblib.load(os.path.join(BASE, "y_train.pkl"))
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

# ---------------------------
# Build an artificial sample
# ---------------------------
# Work directly in the *encoded/scaled* feature space:
rng = np.random.default_rng(11)
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0) + 1e-6
x0 = mu + 0.30 * rng.standard_normal(mu.shape) * sigma  # "invented" point

# Base prediction
proba0 = predict_proba_any(model, kind, x0.reshape(1, -1))
yhat0 = int(np.argmax(proba0, axis=1)[0])

# ---------------------------
# Counterfactual variations
# ---------------------------
# Use SVD of covariance to get top directions of variation
C = np.cov(X_train, rowvar=False) + 1e-6 * np.eye(X_train.shape[1])
U, s, Vt = np.linalg.svd(C)
dirs = Vt[:3]  # top-3 principal directions

scenarios = [("baseline", x0, yhat0, proba0[0].tolist())]
for i, d in enumerate(dirs, start=1):
    for sign, mult in [("+", 1.0), ("-", -1.0)]:
        x_mod = x0 + mult * 0.75 * d * sigma
        p = predict_proba_any(model, kind, x_mod.reshape(1, -1))
        yhat = int(np.argmax(p, axis=1)[0])
        scenarios.append((f"dir{i}{sign}", x_mod, yhat, p[0].tolist()))

# ---------------------------
# Save results
# ---------------------------
rows = []
for tag, vec, yhat, proba in scenarios:
    row = {"Scenario": tag, "Predicted_Class": yhat}
    for k, pk in enumerate(proba):
        row[f"P{k}"] = float(pk)
    rows.append(row)

out = pd.DataFrame(rows)
out_csv = os.path.join(BASE, "point5_artificial_results.csv")
out.to_csv(out_csv, index=False)

# Also save the vectors to allow reproducing each scenario exactly
np.save(os.path.join(BASE, "point5_vectors.npy"),
        np.vstack([v for (_, v, _, _) in scenarios]))

print("=" * 70)
print("POINT 5 – ARTIFICIAL SAMPLE TEST")
print("=" * 70)
print(f"Selected model: {selected_model_name}")
print(f"Results saved to: {out_csv}")
print("\nPreview:")
print(out.head(10).to_string(index=False))
print("\nNote: 'dir1±/dir2±/dir3±' are counterfactual perturbations along top variance directions.")