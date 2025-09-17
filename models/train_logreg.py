"""
Logistic Regression with GridSearchCV for Academic Stress Dataset
----------------------------------------------------------------
This script:
- Loads preprocessed data
- Runs GridSearchCV to tune Logistic Regression
- Evaluates on train/val/test
- Saves the best model
"""

import joblib
from sklearn.linear_model import LogisticRegression
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
# Define model and parameter grid
# ==========================
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"],
    "max_iter": [5000]
}

logreg = LogisticRegression(multi_class="multinomial")

grid_search = GridSearchCV(
    estimator=logreg,
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,
    verbose=2,
    n_jobs=-1
)

# ==========================
# Train with GridSearch
# ==========================
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# ==========================
# Evaluate on train/val/test
# ==========================
y_train_pred = best_model.predict(X_train)
y_val_pred = best_model.predict(X_val)
y_test_pred = best_model.predict(X_test)

print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# ==========================
# Save best model
# ==========================
joblib.dump(best_model, "logreg_best.pkl")
print("Best Logistic Regression model saved as logreg_best.pkl")