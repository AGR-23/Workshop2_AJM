# models/train_gb.py
"""
Train and evaluate Gradient Boosting Classifier
"""

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from preprocessing import get_data

# Load preprocessed data
X_train, X_val, X_test, y_train, y_val, y_test = get_data()

# Define model
gb = GradientBoostingClassifier(random_state=42)

# Small grid for tuning
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0]
}

# Grid search
grid = GridSearchCV(gb, param_grid, cv=3, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

# Best model
best_gb = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# Evaluate
y_train_pred = best_gb.predict(X_train)
y_val_pred = best_gb.predict(X_val)
y_test_pred = best_gb.predict(X_test)

print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# Save model
joblib.dump(best_gb, "gb_model.pkl")
print("Gradient Boosting model saved as gb_model.pkl")