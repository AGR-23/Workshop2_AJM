# ==============================================
# Exploratory Data Analysis (EDA) - Student Stress Dataset
# ==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("academic Stress level - maintainance 1.csv")

# Ensure target is integer
df["stress_index"] = df["stress_index"].astype(int)

# -------------------------------
# 1. Descriptive Statistics
# -------------------------------
print("Descriptive statistics:")
print(df.describe(include="all"))

# -------------------------------
# 2. Distributions of Variables
# -------------------------------
# Numeric/ordinal columns (adjust names according to your dataset)
num_cols = [
    "Peer pressure",
    "Academic pressure from your home",
    "What would you rate the academic competition in your student life",
    "stress_index"
]

# Plot distributions
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[col], palette="viridis")
    plt.title(f"Distribution of {col}")
    plt.show()

# Categorical variables
cat_cols = [
    "Your Academic Stage",
    "Study Environment",
    "What coping strategy you use as a student?",
    "Do you have any bad habits like smoking, drinking on a daily basis?"
]

for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(x=df[col], order=df[col].value_counts().index, palette="muted")
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()

# -------------------------------
# 3. Correlation Matrix
# -------------------------------
plt.figure(figsize=(6,5))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix (Numeric/Ordinal Variables)")
plt.show()

# -------------------------------
# 4. Relationship with Target (stress_index)
# -------------------------------
# Numeric vs target
for col in num_cols[:-1]:  # exclude stress_index itself
    plt.figure(figsize=(6,4))
    sns.boxplot(x="stress_index", y=col, data=df, palette="coolwarm")
    plt.title(f"{col} vs Stress Index")
    plt.show()

# Categorical vs target
for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(x=col, hue="stress_index", data=df, 
                  order=df[col].value_counts().index, palette="Spectral")
    plt.title(f"{col} vs Stress Index")
    plt.xticks(rotation=45)
    plt.show()