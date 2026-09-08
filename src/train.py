# ============================================================

# Hospital Readmission Risk Prediction for Chronic Disease Patients

# Model Training Pipeline

# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score
)

# ============================================================

# Paths

# ============================================================

BASE_DIR = os.path.dirname(
os.path.dirname(os.path.abspath(**file**))
)

DATA_PATH = os.path.join(
BASE_DIR,
"data",
"hospital_readmission_dataset.csv"
)

MODEL_DIR = os.path.join(
BASE_DIR,
"models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================

# Load Dataset

# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# ============================================================

# Basic Cleaning

# ============================================================

df = df.copy()

df = df.dropna(
subset=["Readmitted_30_Days"]
)

# ============================================================

# Define Target

# ============================================================

target = "Readmitted_30_Days"

X = df.drop(
columns=[target]
)

# Remove patient identifier if present

if "Patient_ID" in X.columns:
X = X.drop(
columns=["Patient_ID"]
)

y = df[target]

# ============================================================

# Train-Test Split

# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42,
stratify=y
)

# ============================================================

# Identify Feature Types

# ============================================================

numeric_features = X.select_dtypes(
include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
include=["object", "category"]
).columns.tolist()

# ============================================================

# Preprocessing

# ============================================================

numeric_transformer = Pipeline(
steps=[
(
"imputer",
SimpleImputer(strategy="median")
),
(
"scaler",
StandardScaler()
)
]
)

categorical_transformer = Pipeline(
steps=[
(
"imputer",
SimpleImputer(
strategy="most_frequent"
)
),
(
"encoder",
OneHotEncoder(
handle_unknown="ignore"
)
)
]
)

preprocessor = ColumnTransformer(
transformers=[
(
"num",
numeric_transformer,
numeric_features
),
(
"cat",
categorical_transformer,
categorical_features
)
]
)

# ============================================================

# Models

# ============================================================

models = {

```
"Logistic Regression":
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

"Random Forest":
    RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42
    ),

"Gradient Boosting":
    GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
```

}

# ============================================================

# Train and Evaluate

# ============================================================

results = []

trained_models = {}

for name, classifier in models.items():

```
print(
    f"\nTraining {name}..."
)

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            classifier
        )
    ]
)

pipeline.fit(
    X_train,
    y_train
)

y_pred = pipeline.predict(
    X_test
)

y_prob = pipeline.predict_proba(
    X_test
)[:, 1]

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

results.append({
    "Model": name,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC AUC": roc_auc
})

trained_models[name] = pipeline

print(
    f"{name} ROC-AUC: {roc_auc:.4f}"
)
```

# ============================================================

# Model Comparison

# ============================================================

results_df = pd.DataFrame(
results
)

print(
"\nModel Performance:"
)

print(
results_df.to_string(
index=False
)
)

# ============================================================

# Select Best Model

# ============================================================

best_model_name = (
results_df
.sort_values(
by="ROC AUC",
ascending=False
)
.iloc[0]["Model"]
)

best_model = trained_models[
best_model_name
]

print(
f"\nBest Model: {best_model_name}"
)

# ============================================================

# Save Best Model

# ============================================================

model_path = os.path.join(
MODEL_DIR,
"best_readmission_model.pkl"
)

joblib.dump(
best_model,
model_path
)

print(
f"Model saved to: {model_path}"
)

# ============================================================

# Save Model Results

# ============================================================

results_path = os.path.join(
MODEL_DIR,
"model_comparison.csv"
)

results_df.to_csv(
results_path,
index=False
)

print(
f"Model comparison saved to: {results_path}"
)

print("\nTraining complete.")
