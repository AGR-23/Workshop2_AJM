import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================
# Load preprocessed data
# ==========================
X_train = joblib.load("X_train.pkl")
X_val = joblib.load("X_val.pkl")
X_test = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_val = joblib.load("y_val.pkl")
y_test = joblib.load("y_test.pkl")

# ==========================
# Train model
# ==========================
rf = RandomForestClassifier(
    
    n_estimators=300,
    max_depth=15,  # Ajustar según resultados de búsqueda
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    bootstrap=True,
    
)
rf.fit(X_train, y_train)

# ==========================
# Predictions
# ==========================
y_train_pred = rf.predict(X_train)
y_val_pred = rf.predict(X_val)
y_test_pred = rf.predict(X_test)

# ==========================
# Metrics
# ==========================

from sklearn.model_selection import cross_val_score
import numpy as np



'''
# Probar diferentes profundidades
max_depths = [5, 10, 15, 20, 25, 30, None]
depth_scores = []

for depth in max_depths:
    rf = RandomForestClassifier(
        max_depth=depth,
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    depth_scores.append(scores.mean())
    print(f"Depth {depth}: {scores.mean():.4f}")

# Encontrar la mejor profundidad
best_depth = max_depths[np.argmax(depth_scores)]
print(f"\nMejor profundidad: {best_depth}")'''


results = {
    "Train": {
        "Accuracy": accuracy_score(y_train, y_train_pred),
        "Precision": precision_score(y_train, y_train_pred, average="macro"),
        "Recall": recall_score(y_train, y_train_pred, average="macro"),
        "F1": f1_score(y_train, y_train_pred, average="macro")
    },
    "Validation": {
        "Accuracy": accuracy_score(y_val, y_val_pred),
        "Precision": precision_score(y_val, y_val_pred, average="macro"),
        "Recall": recall_score(y_val, y_val_pred, average="macro"),
        "F1": f1_score(y_val, y_val_pred, average="macro")
    },
    "Test": {
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred, average="macro"),
        "Recall": recall_score(y_test, y_test_pred, average="macro"),
        "F1": f1_score(y_test, y_test_pred, average="macro")
    }
}

results_df = pd.DataFrame(results).T
print(results_df)
