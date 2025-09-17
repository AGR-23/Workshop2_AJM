"""
Random Forest with Hyperparameter Tuning
-----------------------------------------
This script trains a Random Forest classifier on the preprocessed
student stress dataset using GridSearchCV to find the best parameters.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

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
# Define parameter grid
# ==========================
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt", "log2"]
}

# ==========================
# GridSearchCV
# ==========================
rf = RandomForestClassifier(class_weight="balanced", random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_

# ==========================
# Evaluate performance
# ==========================
y_train_pred = best_rf.predict(X_train)
y_val_pred = best_rf.predict(X_val)
y_test_pred = best_rf.predict(X_test)

print("Best Parameters:", grid_search.best_params_)
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# ==========================
# Save model
# ==========================
joblib.dump(best_rf, "random_forest_tuned.pkl")
print("Tuned Random Forest model saved as random_forest_tuned.pkl")