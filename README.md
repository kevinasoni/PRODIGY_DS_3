# PRODIGY_DS_3
# Bank Marketing Purchase Prediction using Decision Tree

## Project Overview

This project builds a **Decision Tree Classifier** to predict whether a bank customer will purchase (subscribe to) a term deposit based on **demographic, behavioral, and socio-economic data**. The model is trained using the **Bank Marketing dataset** from the UCI Machine Learning Repository.

The goal is to demonstrate a complete machine learning workflow including data preprocessing, model training, evaluation, and interpretation.

---

## Problem Statement

Banks run marketing campaigns to encourage customers to subscribe to term deposits. Predicting customer response in advance helps optimize campaign costs and improve conversion rates.

**Objective:**
Predict whether a customer will subscribe to a term deposit (`yes` or `no`) using historical campaign data.

---

## Dataset Description

* **Source:** UCI Machine Learning Repository – Bank Marketing Dataset
* **File Used:** `bank-additional-full.csv`
* **Number of Records:** 41,188
* **Number of Features:** 20 input features + 1 target variable

### Target Variable

* `y`:

  * `yes` → Customer subscribed (Purchase)
  * `no` → Customer did not subscribe

### Feature Types

* **Demographic:** age, job, marital status, education
* **Behavioral:** campaign, previous contacts, contact type
* **Economic Indicators:** euribor3m, employment variation rate, consumer confidence index

---

## Methodology

1. Load and explore the dataset
2. Encode categorical variables using One-Hot Encoding
3. Split data into training (80%) and testing (20%) sets
4. Train a Decision Tree Classifier
5. Evaluate the model using accuracy, confusion matrix, and classification report

---

## Model Details

* **Algorithm:** Decision Tree Classifier
* **Criterion:** Gini Index
* **Max Depth:** 6
* **Min Samples Split:** 50

These parameters help control overfitting while maintaining interpretability.

---

## Results

* **Accuracy:** ~92%
* The model performs strongly in identifying customers who are unlikely to subscribe.
* Purchase prediction is more challenging due to class imbalance, which is typical in marketing datasets.

---

## Repository Structure

```
bank-marketing-decision-tree/
│
├── data/
│   └── bank-additional-full.csv
│
├── src/
│   └── decision_tree_model.py
│
├── notebooks/
│   └── decision_tree_bank_marketing.ipynb
│
├── results/
│   └── evaluation_metrics.txt
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run the Project

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Python script:

   ```bash
   python src/decision_tree_model.py
   ```

---

## Future Improvements

* Handle class imbalance using SMOTE or class weights
* Perform feature selection
* Compare performance with Random Forest or Logistic Regression
* Exclude `duration` feature for a strictly realistic pre-call model

---

## References

Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems.
