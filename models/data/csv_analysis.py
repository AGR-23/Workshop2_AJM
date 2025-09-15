# Inspect dataset: head, dtypes, missing, uniques, suggest target and variable types
import pandas as pd
import numpy as np

FILEPATH = "academic Stress level - maintainance 1.csv"
df = pd.read_csv(FILEPATH)

print("Shape:", df.shape)
print("\nHead:")
print(df.head())

print("\nColumn dtypes and uniques:")
for c in df.columns:
    s = df[c]
    print(f"- {c!r}: dtype={s.dtype}, n_unique={s.nunique(dropna=True)}, n_missing={s.isna().sum()}, examples={s.dropna().unique()[:6].tolist()}")

# Heuristic to find target candidates
candidates = [c for c in df.columns if "stress" in c.lower() or ("rate" in c.lower() and "stress" in c.lower())]
if not candidates:
    candidates = [c for c in df.columns if "level" in c.lower() and df[c].nunique()<10]
print("\nSuggested target columns:", candidates)

# Decide task by target characteristics
if candidates:
    target = candidates[0]
    print("\nSelected target:", repr(target))
    print("dtype:", df[target].dtype, "n_unique:", df[target].nunique())
    # Heuristic: if target numeric with few unique -> classification (ordinal); if many unique -> regression
    if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() <= 20:
        task = "classification (ordinal / multi-class)"
    else:
        task = "regression"
    print("Heuristic task decision:", task)
else:
    print("No clear target found; please indicate target column name.")