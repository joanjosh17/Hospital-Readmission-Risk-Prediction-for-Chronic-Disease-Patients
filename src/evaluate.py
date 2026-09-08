# ============================================================

# Hospital Readmission Risk Prediction

# Model Evaluation

# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
classification_report,
confusion_matrix,
roc_auc_score,
roc_curve
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

MODEL_PATH = os.path.join(
BASE_DIR,
"models",
"best_readmission_model.pkl"
)

CHARTS_DIR = os.path.join(
BASE_DIR,
"charts"
)

os.makedirs(
CHARTS_DIR,
exist_ok=True
)

# ============================================================

# Load Data

# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
DATA_PATH
)

target = "Readmitted_30_Days"

X = df.drop(
columns=[target]
)

if "Patient_ID" in X.columns:
X = X.drop(
columns=["Patient_ID"]
)

y = df[target]

# ============================================================

# Recreate Test Split

# ============================================================

_, X_test, _, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42,
stratify=y
)

# ============================================================

# Load Model

# ============================================================

print("Loading trained model...")

model = joblib.load(
MODEL_PATH
)

# ============================================================

# Predictions

# ============================================================

y_pred = model.predict(
X_test
)

y_prob = model.predict_proba(
X_test
)[:, 1]

# ============================================================

# Metrics

# ============================================================

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

print("\n======================================")
print("MODEL EVALUATION")
print("======================================")

print(
f"Accuracy  : {accuracy:.4f}"
)

print(
f"Precision : {precision:.4f}"
)

print(
f"Recall    : {recall:.4f}"
)

print(
f"F1 Score  : {f1:.4f}"
)

print(
f"ROC AUC   : {roc_auc:.4f}"
)

# ============================================================

# Classification Report

# ============================================================

print(
"\nClassification Report:"
)

print(
classification_report(
y_test,
y_pred,
zero_division=0
)
)

# ============================================================

# Save Metrics

# ============================================================

metrics_df = pd.DataFrame({
"Metric": [
"Accuracy",
"Precision",
"Recall",
"F1 Score",
"ROC AUC"
],
"Score": [
accuracy,
precision,
recall,
f1,
roc_auc
]
})

metrics_df.to_csv(
os.path.join(
CHARTS_DIR,
"evaluation_metrics.csv"
),
index=False
)

# ============================================================

# Confusion Matrix

# ============================================================

cm = confusion_matrix(
y_test,
y_pred
)

plt.figure(
figsize=(7, 6)
)

sns.heatmap(
cm,
annot=True,
fmt="d",
cmap="Blues",
xticklabels=[
"Not Readmitted",
"Readmitted"
],
yticklabels=[
"Not Readmitted",
"Readmitted"
]
)

plt.title(
"Hospital Readmission Prediction - Confusion Matrix"
)

plt.xlabel(
"Predicted"
)

plt.ylabel(
"Actual"
)

plt.tight_layout()

plt.savefig(
os.path.join(
CHARTS_DIR,
"confusion_matrix.png"
),
dpi=300
)

plt.close()

# ============================================================

# ROC Curve

# ============================================================

fpr, tpr, _ = roc_curve(
y_test,
y_prob
)

plt.figure(
figsize=(8, 6)
)

plt.plot(
fpr,
tpr,
linewidth=2,
label=f"Model (AUC = {roc_auc:.3f})"
)

plt.plot(
[0, 1],
[0, 1],
linestyle="--",
label="Random Classifier"
)

plt.xlabel(
"False Positive Rate"
)

plt.ylabel(
"True Positive Rate"
)

plt.title(
"ROC Curve - Hospital Readmission Prediction"
)

plt.legend()

plt.tight_layout()

plt.savefig(
os.path.join(
CHARTS_DIR,
"roc_curve.png"
),
dpi=300
)

plt.close()

print(
"\nEvaluation completed successfully."
)
