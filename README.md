# 💳 Credit Card Fraud Detection (Machine Learning Model)

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?logo=scikitlearn)](https://scikit-learn.org/)

> Detecting fraudulent credit card transactions using advanced machine learning techniques on an imbalanced dataset.  
> This project helps financial institutions and businesses prevent unauthorized transactions and minimize losses.

---

## 📘 Overview

It is crucial for credit card companies to recognize fraudulent transactions so that customers are not charged for items they did not purchase.

This project focuses on building a **Credit Card Fraud Detection** model using real-world transaction data. The dataset contains transactions made by European cardholders in **September 2013**, where **492 out of 284,807 transactions** were fraudulent — accounting for only **0.172%** of all transactions.

---

## 🧩 Dataset Description

- **Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Transactions:** 284,807  
- **Frauds:** 492 (0.172%)  
- **Data Type:** Numerical (result of PCA transformation)
- **Features:**
  - `V1` to `V28`: Principal Components (PCA-transformed features)
  - `Time`: Seconds elapsed between each transaction and the first transaction
  - `Amount`: Transaction amount (useful for cost-sensitive learning)
  - `Class`: Target variable (`1` = Fraud, `0` = Legitimate)

Due to confidentiality, original features are not disclosed.  
Evaluation metrics like **AUC-PR (Area Under the Precision-Recall Curve)** are recommended since the dataset is highly imbalanced.

---

## 🧠 Features

- Handles **highly imbalanced dataset** effectively.  
- Implements **data preprocessing and scaling**.  
- Compares multiple models: Logistic Regression, Random Forest, XGBoost, etc.  
- Evaluates using **AUC-PR, F1-score**, and **Confusion Matrix**.  
- Visualizes fraud detection patterns and feature importance.  

---

## 🛠️ Tech Stack

| Component | Technology |
|------------|-------------|
| Programming | Python |
| Libraries | NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn |
| Model | Logistic Regression, Random Forest, XGBoost |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook / Streamlit (optional for deployment) |

---

