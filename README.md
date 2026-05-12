# Hospital Readmission Risk Prediction for Chronic Disease Patients

## Overview

This project predicts **30-day hospital readmission risk** for chronic disease patients using synthetic healthcare data and machine learning models. It identifies high-risk patients based on clinical, demographic, and behavioral factors.

---

## Dataset

- 15,000 synthetic patient records
- Chronic diseases: Diabetes, Hypertension, Heart Disease, COPD, CKD
- Target: `Readmitted_30_Days`

---

## Machine Learning Models

- Logistic Regression  
- Random Forest  
- Gradient Boosting  

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.747 | 0.747 | 1.000 | 0.855 | 0.634 |
| Random Forest | 0.745 | 0.747 | 0.995 | 0.853 | 0.598 |
| Gradient Boosting | 0.745 | 0.747 | 0.996 | 0.854 | 0.639 |

---

## Feature Importance

Top predictors of readmission:
- Medication Adherence
- BMI
- Glucose Level
- Blood Pressure
- Heart Rate
- Age
- Length of Stay

---

# Visualizations

All charts are stored in:

```
Hospital-Readmission-Risk-Prediction-for-Chronic-Disease-Patients/charts/
```

---

## 📊 1. Readmission Distribution

![Figure 1](Hospital-Readmission-Risk-Prediction-for-Chronic-Disease-Patients/charts/Figure_1.png)

---

## 📊 2. Age Distribution

![Figure 2](Hospital-Readmission-Risk-Prediction-for-Chronic-Disease-Patients/charts/Figure_2.png)

---

## 📊 3. Length of Stay vs Readmission

![Figure 3](Hospital-Readmission-Risk-Prediction-for-Chronic-Disease-Patients/charts/Figure_3.png)

---

## 📊 4. Medication Adherence vs Readmission

![Figure 4](Hospital-Readmission-Risk-Prediction-for-Chronic-Disease-Patients/charts/Figure_4.png)

---

## 📊 5. Confusion Matrix (Logistic Regression)

![Figure 5](Hospital-Readmission-Risk-Prediction-for-Chronic-Disease-Patients/charts/Figure_5.png)

---

## 📊 6. Confusion Matrix (Random Forest)

![Figure 6](Hospital-Readmission-Risk-Prediction-for-Chronic-Disease-Patients/charts/Figure_6.png)

---

## 📊 7. Confusion Matrix (Gradient Boosting)

![Figure 7](Hospital-Readmission-Risk-Prediction-for-Chronic-Disease-Patients/charts/Figure_7.png)

---

## 📊 8. Model ROC-AUC Comparison

![Figure 8](Hospital-Readmission-Risk-Prediction-for-Chronic-Disease-Patients/charts/Figure_8.png)

---

## 📊 9. Feature Importance

![Figure 9](Hospital-Readmission-Risk-Prediction-for-Chronic-Disease-Patients/charts/Figure_9.png)

---

## Project Structure

```
Hospital-Readmission-Risk-Prediction-for-Chronic-Disease-Patients/
│
├── readmission.py
├── hospital_readmission_dataset.csv
├── README.md
│
└── charts/
    ├── Figure_1.png
    ├── Figure_2.png
    ├── Figure_3.png
    ├── Figure_4.png
    ├── Figure_5.png
    ├── Figure_6.png
    ├── Figure_7.png
    ├── Figure_8.png
    └── Figure_9.png
```

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python readmission.py
```

---

## Key Insights

- Medication adherence is the strongest predictor of readmission
- High glucose levels and BMI increase risk
- Longer hospital stays correlate with readmission
- Chronic disease type significantly impacts outcomes

---

## Future Improvements

- XGBoost / LightGBM models
- SHAP explainability
- Streamlit dashboard
- Real hospital dataset integration

---

## Author

**Joan Joshua**
