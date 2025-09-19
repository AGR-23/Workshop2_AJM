"""
Random Forest with Cross Validation and Comparative Table , where the fuck is the comparative table I don't see it 
---------------------------------------------------------
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
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
# Combine train + validation for CV
# ==========================
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

# ==========================
# Setup Cross-Validation (5-fold)
# ==========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=15,
    min_samples_leaf=8,
    max_features=0.6,
    random_state=42,
    n_jobs=-1
)

# ==========================
# PERFORM CROSS VALIDATION - ESTO ES LO NUEVO
# ==========================
print("Performing 5-fold Cross-Validation...")
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv_results = cross_validate(rf, X_train_full, y_train_full, 
                           cv=cv, scoring=scoring, n_jobs=-1)

print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS (5-fold)")
print("="*60)
print(f"CV Accuracy:  {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std()*2:.4f})")
print(f"CV Precision: {cv_results['test_precision_weighted'].mean():.4f} (+/- {cv_results['test_precision_weighted'].std()*2:.4f})")
print(f"CV Recall:    {cv_results['test_recall_weighted'].mean():.4f} (+/- {cv_results['test_recall_weighted'].std()*2:.4f})")
print(f"CV F1:        {cv_results['test_f1_weighted'].mean():.4f} (+/- {cv_results['test_f1_weighted'].std()*2:.4f})")

# ==========================
# Train final model on full training data
# ==========================
rf.fit(X_train_full, y_train_full)

# ==========================
# Predictions for all sets
# ==========================
y_train_pred = rf.predict(X_train)
y_val_pred = rf.predict(X_val)
y_test_pred = rf.predict(X_test)

# ==========================
# Calculate all metrics
# ==========================
def calculate_metrics(y_true, y_pred, set_name):
    return {
        'Set': set_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

metrics_train = calculate_metrics(y_train, y_train_pred, 'X_train')
metrics_val = calculate_metrics(y_val, y_val_pred, 'X_val')
metrics_test = calculate_metrics(y_test, y_test_pred, 'X_test')

# ==========================
# Create comparative table
# ==========================
comparative_table = pd.DataFrame([metrics_train, metrics_val, metrics_test])
comparative_table = comparative_table.set_index('Set')

# ==========================
# Print results
# ==========================
print("\n" + "="*80)
print("COMPARATIVE TABLE - RANDOM FOREST")
print("="*80)
print(comparative_table.round(4))

print("\n" + "="*80)
print("OVERFITTING ANALYSIS")
print("="*80)
overfitting_gap = comparative_table.loc['X_train', 'Accuracy'] - comparative_table.loc['X_val', 'Accuracy']
generalization_gap = comparative_table.loc['X_train', 'Accuracy'] - comparative_table.loc['X_test', 'Accuracy']

print(f"Train-Val gap: {overfitting_gap:.4f}")
print(f"Train-Test gap: {generalization_gap:.4f}")

# ==========================
# Save results and model
# ==========================
comparative_table.to_csv("random_forest_comparative_table.csv") #if needed erase the # and run the code
joblib.dump(rf, "random_forest_model.pkl")

print("\n✅ Comparative table: random_forest_comparative_table.csv")
print("✅ Trained model: random_forest_final_model.pkl")