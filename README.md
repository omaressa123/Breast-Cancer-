# Breast Cancer Classification App

This repository contains a machine learning project for classifying breast cancer tumors as **Benign** or **Malignant** using a **Random Forest Classifier**.  
It includes data exploration, model training, evaluation, and a simple web interface to make predictions.

---

## 1. Introduction
Breast cancer is one of the most common cancers worldwide. Early and accurate detection can help improve treatment outcomes.  
This project uses the **Breast Cancer Wisconsin dataset** and applies machine learning techniques to predict whether a tumor is benign (B) or malignant (M).

---

## 2. Contents
- `breast_cancer_dataset.csv` – Dataset used for training and testing.  
- `breast-cancer.ipynb` – Jupyter Notebook for data exploration, feature engineering, training, and evaluation.  
- `breast_cancer_rf_model.pkl` – Trained Random Forest model saved in Pickle format.  
- `app.py` – Web application for real-time predictions.  
- `requirements.txt` – List of dependencies.  

---

## 3. Requirements
- Python 3.x  
- Install the dependencies:
```bash
pip install -r requirements.txt

[ User Input ] 
      ↓
[ Preprocessing ]
      ↓
[ Random Forest Model ]
      ↓
[ Prediction → Benign / Malignant ]
