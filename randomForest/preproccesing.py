"""
Data Preprocessing Pipeline for Student Academic Stress Dataset
---------------------------------------------------------------
This script handles data cleaning, categorical encoding, feature scaling, 
and dataset splitting into train/validation/test sets.

Steps:
1. Handle missing values
2. Encode categorical variables
3. Normalize/standardize numerical variables
4. Build a preprocessing pipeline with scikit-learn
5. Split dataset into X_train, X_val, X_test, y_train, y_val, y_test (70/15/15)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# ==========================
# Load dataset
# ==========================
df = pd.read_csv("student_academic_stress.csv")

# Rename target for consistency
df = df.rename(columns={"Rate your academic stress index ": "stress_index"})

# ==========================
# Separate features and target
# ==========================
X = df.drop(columns=["stress_index", "Timestamp"])  # Drop target and timestamp
y = df["stress_index"]

# ==========================
# Identify categorical and numerical features
# ==========================
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

# ==========================
# Define preprocessing steps
# ==========================
# For categorical features: impute missing values + one-hot encoding
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# For numerical features: impute missing values + standard scaling
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Combine both transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_transformer, categorical_features),
        ("numerical", numerical_transformer, numerical_features)
    ]
)

# ==========================
# Final pipeline
# ==========================
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# ==========================
# Split dataset (70/15/15)
# ==========================
# First split: train (70%) + temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Second split: validation (15%) + test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# ==========================
# Fit pipeline only on training data
# ==========================
pipeline.fit(X_train)

# Apply transformations
X_train_processed = pipeline.transform(X_train)
X_val_processed = pipeline.transform(X_val)
X_test_processed = pipeline.transform(X_test)

# ==========================
# Export processed data
# ==========================
def get_data():
    """
    Returns the preprocessed training, validation, and test sets.
    """
    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test

# Save processed data
joblib.dump(X_train_processed, "X_train.pkl")
joblib.dump(X_val_processed, "X_val.pkl")
joblib.dump(X_test_processed, "X_test.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_val, "y_val.pkl")
joblib.dump(y_test, "y_test.pkl")

joblib.dump(pipeline, "preprocessing_pipeline.pkl")

print("Data preprocessing completed. Files saved as .pkl")