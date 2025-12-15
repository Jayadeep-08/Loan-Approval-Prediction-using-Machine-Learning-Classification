# 🏦 Loan Approval Prediction using Machine Learning Classification

Predict whether a loan application should be **approved or rejected** using supervised machine learning on structured financial and demographic data.  
The project compares **Logistic Regression** and **Decision Tree Classification**, achieving an accuracy of ~88% with a tuned Decision Tree on ~50k records.

---

## 🚀 Project Snapshot

- **Goal:** Predict loan eligibility based on applicant attributes.
- **Problem Type:** Binary Classification.
- **Best Model:** Decision Tree Classifier (tuned).
- **Final Performance:** Accuracy ≈ 0.88, strong ROC-AUC.
- **Dataset Size:** ~50,000 loan applications.
- **Core Idea:** Evaluate linear vs non-linear classifiers with proper preprocessing and risk-aware evaluation.

---

## 📌 Problem Statement

Loan approval decisions involve multiple interacting factors such as income, credit score, debt burden, employment status, and loan intent.  
Manual decision processes are time-consuming and inconsistent.

This project aims to:

- Automate loan approval decisions
- Compare **linear** and **non-linear** classification models
- Evaluate decisions using **risk-oriented metrics** relevant to banking

---

## 📂 Dataset Overview

- **File:** `Loan_approval_data_2025.csv`
- **Records:** 50,000
- **Target Variable:**  
  - `loan_status`
    - `1` → Approved
    - `0` → Rejected

### Feature Categories

**Numerical Features**
- Age
- Annual Income
- Credit Score
- Loan Amount
- Debt-to-Income Ratio
- Loan-to-Income Ratio
- Credit History Years

**Categorical Features**
- Occupation Status
- Loan Intent
- Loan Type

---

## 🔍 Data Processing Pipeline

### Step 1: Dataset Inspection

The dataset is loaded using Pandas and inspected for:
- Missing values
- Data types
- Irrelevant identifiers

The `customer_id` column is removed since it does not contribute predictive value.

> 🖼️ **Dataset Preview**  
> <img width="819" height="828" alt="image" src="https://github.com/user-attachments/assets/357edf71-517f-4cc9-a09c-2bea687fa49d" />


---

### Step 2: Feature Type Identification

Features are split into:
- **Categorical columns** → require encoding
- **Numerical columns** → require scaling

This separation is essential to ensure correct preprocessing for each data type.

---

### Step 3: Feature Transformation

#### Categorical Encoding
- Applied **One-Hot Encoding**
- Converts categories into binary vectors
- Prevents false ordinal relationships

#### Numerical Scaling
- Applied **Standard Scaling**
- Normalizes feature distributions
- Essential for gradient-based models like Logistic Regression

A `ColumnTransformer` ensures transformations are applied consistently.

> 🖼️ **Preprocessing Pipeline**  
><img width="747" height="117" alt="image" src="https://github.com/user-attachments/assets/058c58a4-1882-41ba-9940-2131e66c6b9d" />


---

## 🧠 Models Implemented

### 1️⃣ Logistic Regression (Baseline Model)

- Linear classification model
- Outputs probability using sigmoid function
- Assumes linear decision boundary
- Configured with:
  - Increased iteration limit
  - Balanced class weights

**Observed Performance**
- Accuracy ≈ 80%
- Acts as a strong baseline but underfits complex patterns

> 🖼️ **Logistic Regression Output**  
> <img width="456" height="152" alt="image" src="https://github.com/user-attachments/assets/d7af9886-55a6-421d-abe3-4677cdff62ec" />


---

### 2️⃣ Decision Tree Classifier (Final Model)

- Non-linear, rule-based classifier
- Captures complex feature interactions
- Decision paths resemble human reasoning (if-else rules)

**Hyperparameter Tuning**
- Performed using `GridSearchCV`
- Tuned parameters:
  - `max_depth`
  - `min_samples_leaf`

This prevents overfitting while retaining predictive power.

> 🖼️ **Decision Tree Structure (Conceptual)**  
> <img width="401" height="106" alt="image" src="https://github.com/user-attachments/assets/abcd378a-fd43-4a8e-9d44-e2f4b6d2a9ce" />


---

## 🧪 Training and Validation Strategy

- **Train-Test Split:** 80% / 20%
- **Stratified Sampling:** Maintains approval/rejection ratio
- **Cross-Validation:** Used during Decision Tree tuning

This ensures unbiased and reliable evaluation.

---

## 📊 Evaluation Metrics

The models are evaluated using:

- **Accuracy** – Overall correctness
- **ROC-AUC** – Class separation ability
- **Precision** – Reliability of approved loans
- **Recall** – Coverage of eligible applicants
- **F1-Score** – Balance between precision and recall

---

## 📈 Confusion Matrix (Matrix-Based Graph)
            Predicted
         Rejected   Approved
Actual Rejected TN FP
Actual Approved FN TP


### Banking Interpretation

- **False Positives (FP):** Risky approvals (most critical error)
- **False Negatives (FN):** Lost potential customers

Separate confusion matrices are generated for:
- Logistic Regression
- Decision Tree

> 🖼️ **Confusion Matrix – Logistic Regression**  
> <img width="796" height="613" alt="image" src="https://github.com/user-attachments/assets/6788f67d-d504-4b40-9562-1270a8335dfd" />


> 🖼️ **Confusion Matrix – Decision Tree**  
> <img width="796" height="603" alt="image" src="https://github.com/user-attachments/assets/136572bf-b7cc-417e-b9a5-0e95ddd52070" />


---

## 📊 Performance Summary

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|--------|
| Logistic Regression | ~0.80    | Moderate |
| Decision Tree       | ~0.88    | High |

The **Decision Tree outperforms Logistic Regression**, indicating strong non-linear relationships in the data.

---

## 💾 Model Persistence

Trained models are saved using `joblib`:

- `logistic_model.joblib`
- `decision_tree_model.joblib`

This allows reuse without retraining and supports deployment.

---

## 🧩 Key Concepts Demonstrated

- Binary Classification
- Feature-specific preprocessing
- One-Hot Encoding
- Feature Scaling
- Linear vs Non-Linear Models
- Hyperparameter Tuning
- Confusion Matrix Analysis
- Risk-aware model evaluation

---

## 🏁 Conclusion

This project demonstrates a complete **end-to-end machine learning pipeline** for loan approval prediction.  
While Logistic Regression provides a strong linear baseline, the Decision Tree model captures complex decision boundaries and delivers superior performance.

The results highlight that **model choice, preprocessing, and evaluation strategy** are equally important in real-world financial ML systems.

---

## 👤 Author

**N. Jayadeep**  
CSE – Cybersecurity  

_Last updated: December 2025_


A confusion matrix is used to visualize prediction outcomes.

