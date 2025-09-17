import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
from collections import Counter

# ==========================
# Load dataset
# ==========================
df = pd.read_csv("data/academic Stress level - maintainance 1.csv")
df = df.rename(columns={"What is your stress level?": "stress_index"})

# ==========================
# Separate features and target
# ==========================
X = df.drop(columns=["stress_index", "Timestamp"], errors="ignore")
y = df["stress_index"]

# ==========================
# Split dataset (stratified 70/15/15)
# ==========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# ==========================
# Identify feature types
# ==========================
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

# ==========================
# Preprocessing (imputation + encoding)
# ==========================
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Apply only imputation + encoding before SMOTE
pre_smote = ColumnTransformer(
    transformers=[
        ("categorical", categorical_transformer, categorical_features),
        ("numerical", numerical_transformer, numerical_features)
    ]
)

# Fit only on train
pre_smote.fit(X_train)

# Transform all datasets
X_train_enc = pre_smote.transform(X_train)
X_val_enc = pre_smote.transform(X_val)
X_test_enc = pre_smote.transform(X_test)

# ==========================
# Apply SMOTE on encoded data
# ==========================
class_counts = Counter(y_train)
minority_class_size = min(class_counts.values())

if minority_class_size > 1:
    k_neighbors = min(5, minority_class_size - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_enc, y_train)
    print("SMOTE applied:", Counter(y_train_bal))
else:
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_train_bal, y_train_bal = ros.fit_resample(X_train_enc, y_train)
    print("RandomOverSampler applied:", Counter(y_train_bal))

# ==========================
# Scale AFTER balancing
# ==========================
scaler = StandardScaler()
X_train_bal = scaler.fit_transform(X_train_bal)
X_val_enc = scaler.transform(X_val_enc)
X_test_enc = scaler.transform(X_test_enc)

# ==========================
# Export processed data
# ==========================
def get_data():
    return X_train_bal, X_val_enc, X_test_enc, y_train_bal, y_val, y_test

joblib.dump(X_train_bal, "X_train.pkl")
joblib.dump(X_val_enc, "X_val.pkl")
joblib.dump(X_test_enc, "X_test.pkl")
joblib.dump(y_train_bal, "y_train.pkl")
joblib.dump(y_val, "y_val.pkl")
joblib.dump(y_test, "y_test.pkl")

print("Preprocessing finished. Data balanced + scaled after SMOTE.")