# 🌸 Iris Flower Species Classification (Streamlit App)

This is a Streamlit-based machine learning web app for classifying Iris flower species based on sepal and petal measurements.

## 📊 Features

- Uses the **Iris dataset** from `sklearn`
- Supports **Logistic Regression**, **Random Forest**, and **K-Nearest Neighbors**
- Performs:
  - Exploratory Data Analysis (EDA)
  - Model Training & Evaluation
  - Multiclass ROC Curve & AUC Score
  - Cross-Validation Score (CV=5)
  - Live Prediction using sliders

## 📦 Requirements

Create a virtual environment and install dependencies:

```bash
python -m venv venv
.\venv\Scripts\activate   # For Windows
pip install -r requirements.txt
