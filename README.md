# Hospital Readmission Risk Prediction for Chronic Disease Patients

## Overview

Hospital readmissions within 30 days of discharge are an important healthcare quality and resource-management challenge. Identifying patients who are at elevated risk of readmission can support earlier intervention, improved discharge planning, and more targeted follow-up.

This project develops a machine learning framework for predicting **30-day hospital readmission risk among patients with chronic diseases** using demographic, clinical, behavioral, and hospitalization-related factors.

The project uses a **synthetic healthcare dataset containing 15,000 patient records** and evaluates multiple supervised machine learning algorithms to identify patterns associated with hospital readmission.

> **Disclaimer:** This project is intended for educational, research, and demonstration purposes only. It is not a clinical decision-support system and should not be used for real-world medical decision-making.

---

## Objectives

The main objectives of this project are to:

* Predict whether a patient is likely to be readmitted within 30 days.
* Identify important demographic, clinical, and behavioral predictors of readmission.
* Compare the performance of multiple machine learning classification algorithms.
* Evaluate models using clinically relevant classification metrics.
* Visualize patterns and relationships associated with hospital readmission.
* Demonstrate an end-to-end healthcare machine learning workflow.

---

## Dataset

The project uses a synthetic healthcare dataset containing **15,000 patient records**.

The dataset represents patients with chronic health conditions and includes demographic, clinical, behavioral, and hospital-related variables.

### Chronic Disease Categories

The dataset includes conditions such as:

* Diabetes
* Hypertension
* Heart Disease
* Chronic Obstructive Pulmonary Disease (COPD)
* Chronic Kidney Disease (CKD)

### Target Variable

The prediction target is:

```text
Readmitted_30_Days
```

Where:

* `0` = Patient was not readmitted within 30 days
* `1` = Patient was readmitted within 30 days

Because the dataset is synthetic, the results should not be interpreted as representing actual hospital populations or clinical outcomes.

---

## Machine Learning Workflow

The project follows a complete machine learning pipeline:

```text
Synthetic Healthcare Dataset
            │
            ▼
     Data Preparation
            │
            ▼
 Exploratory Data Analysis
            │
            ▼
   Feature Engineering
            │
            ▼
    Train/Test Split
            │
            ▼
     Model Training
            │
      ┌─────┼─────┐
      ▼     ▼     ▼
 Logistic  Random  Gradient
Regression Forest  Boosting
      │     │     │
      └─────┼─────┘
            ▼
      Model Evaluation
            │
            ▼
 Feature Importance Analysis
            │
            ▼
    Readmission Risk
       Prediction
```

---

## Models Evaluated

Three supervised classification algorithms are evaluated.

### 1. Logistic Regression

A linear classification algorithm used as a baseline model for estimating the probability of 30-day readmission.

### 2. Random Forest

An ensemble learning algorithm that combines multiple decision trees to model nonlinear relationships between patient characteristics and readmission risk.

### 3. Gradient Boosting

An ensemble technique that sequentially builds decision trees to improve predictive performance by learning from previous model errors.

---

## Model Performance

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |    0.747 |     0.747 |  1.000 |    0.855 |   0.634 |
| Random Forest       |    0.745 |     0.747 |  0.995 |    0.853 |   0.598 |
| Gradient Boosting   |    0.745 |     0.747 |  0.996 |    0.854 |   0.639 |

The results show that all three models achieve similar classification performance on the synthetic dataset. Gradient Boosting achieves the highest ROC-AUC among the evaluated models, while Logistic Regression provides the highest accuracy, recall, and F1 score in the reported evaluation.

These results are specific to the synthetic dataset and should not be interpreted as evidence of clinical effectiveness.

---

## Key Predictors

Feature analysis identified several variables associated with the model's predictions.

The leading predictors include:

1. Medication Adherence
2. BMI
3. Glucose Level
4. Blood Pressure
5. Heart Rate
6. Age
7. Length of Stay

These variables provide useful analytical signals for understanding the factors represented in the synthetic data that contribute to predicted readmission risk.

---

## Exploratory Data Analysis

The project includes visualizations examining important characteristics of the dataset.

### Readmission Distribution

Shows the distribution of patients according to their 30-day readmission status.

### Age Distribution

Examines the age profile of patients represented in the dataset.

### Length of Stay vs Readmission

Explores the relationship between hospitalization duration and 30-day readmission.

### Medication Adherence vs Readmission

Examines the relationship between medication adherence and readmission outcomes.

---

## Model Evaluation Visualizations

The following model-performance visualizations are included:

* Logistic Regression confusion matrix
* Random Forest confusion matrix
* Gradient Boosting confusion matrix
* ROC-AUC comparison
* Feature importance

All visualization outputs are stored in the:

```text
charts/
```

directory.

---

## Project Structure

```text
Hospital-Readmission-Risk-Prediction-for-Chronic-Disease-Patients/
│
├── data/
│   └── hospital_readmission_dataset.csv
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── charts/
│   ├── readmission_distribution.png
│   ├── age_distribution.png
│   ├── length_of_stay_vs_readmission.png
│   ├── medication_adherence_vs_readmission.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_gradient_boosting.png
│   ├── model_roc_auc_comparison.png
│   └── feature_importance.png
│
├── models/
│   └── best_readmission_model.pkl
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/joanjosh17/Hospital-Readmission-Risk-Prediction-for-Chronic-Disease-Patients.git
```

### 2. Navigate into the project

```bash
cd Hospital-Readmission-Risk-Prediction-for-Chronic-Disease-Patients
```

### 3. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train the models

```bash
python src/train.py
```

The training script performs data preparation, feature engineering, model training, and model comparison.

### Evaluate the trained models

```bash
python src/evaluate.py
```

The evaluation script generates model-performance metrics and visualization outputs.

### Generate a patient risk prediction

```bash
python src/predict.py
```

The prediction script loads the trained model and demonstrates how an individual patient's information can be passed through the prediction pipeline.

---

## Model Evaluation Metrics

### Accuracy

Measures the proportion of predictions that are correctly classified.

### Precision

Measures the proportion of predicted readmissions that are actually readmissions.

### Recall

Measures the proportion of actual readmissions correctly identified by the model.

### F1 Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between patients with and without the target outcome across different classification thresholds.

For healthcare risk prediction, recall can be particularly important when missing a high-risk patient has significant consequences. However, precision, calibration, specificity, and clinical utility should also be considered before any real-world deployment.

---

## Key Insights

The analysis of the synthetic dataset suggests several important patterns:

* Medication adherence is one of the strongest predictive variables.
* BMI and glucose level contribute substantially to predicted readmission risk.
* Blood pressure and heart rate provide important clinical signals.
* Age is associated with variation in predicted readmission risk.
* Longer hospital stays show a relationship with readmission outcomes.
* Chronic disease characteristics influence patient risk profiles.

These findings describe patterns within the synthetic dataset and are not intended to establish clinical causality.

---

## Limitations

This project has several limitations:

1. **Synthetic Data**
   The dataset is artificially generated and does not represent real patient records.

2. **No Clinical Validation**
   The models have not undergone prospective clinical validation.

3. **Potential Synthetic Bias**
   Relationships within synthetic data may not accurately reproduce real-world healthcare relationships.

4. **Limited Model Scope**
   Only three classification algorithms are evaluated.

5. **No External Validation**
   The models have not been tested on an independent real-world dataset.

6. **No Clinical Deployment**
   The model is intended as a data-science demonstration rather than a production clinical system.

---

## Future Improvements

Potential extensions include:

* Hyperparameter optimization
* XGBoost and LightGBM comparison
* SHAP-based model explainability
* Probability calibration
* Cross-validation
* External validation using real-world datasets
* Fairness and subgroup performance analysis
* Cost-sensitive learning
* Streamlit-based risk prediction dashboard
* Model monitoring and drift detection
* REST API deployment
* Containerization with Docker

---

## Ethical Considerations

Machine learning in healthcare requires careful consideration of:

* Patient privacy
* Data security
* Algorithmic bias
* Fairness across demographic groups
* Model interpretability
* Clinical validation
* Human oversight

Predictions from machine learning systems should complement—not replace—qualified clinical judgment.

---

## Reproducibility

The project uses fixed random seeds where applicable to improve reproducibility.

Recommended environment:

```text
Python 3.10+
```

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

## License

This project is licensed under the **MIT License**.

See the `LICENSE` file for the complete license terms.

---

## Author

**Joan Joshua**

GitHub: [@joanjosh17](https://github.com/joanjosh17)

---

## Disclaimer

This project is intended strictly for educational, research, and demonstration purposes.

The machine learning models and predictions generated by this project are **not medical diagnoses, treatment recommendations, or clinical decisions**. The synthetic dataset does not represent real patients, hospitals, or clinical outcomes.

The model should not be used as a substitute for professional medical judgment, validated clinical decision-support systems, or appropriate clinical evaluation.

---

## Acknowledgments

This project demonstrates the application of Python-based data science and machine learning techniques to a public-health and healthcare analytics problem using synthetic data.
