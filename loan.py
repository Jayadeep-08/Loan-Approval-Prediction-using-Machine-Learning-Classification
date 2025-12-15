import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib

warnings.filterwarnings("ignore")

# -------------------------------
# Load dataset (same folder)
# -------------------------------
df = pd.read_csv("Loan_approval_data_2025.csv")

# Drop ID column
if 'customer_id' in df.columns:
    df = df.drop(columns=['customer_id'])

# Target & features
y = df['loan_status']
X = df.drop(columns=['loan_status'])

# Identify column types
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# -------------------------------
# Preprocessing
# -------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', StandardScaler(), num_cols)
    ]
)

# -------------------------------
# Logistic Regression Model
# -------------------------------
log_model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('clf', LogisticRegression(
        solver='saga',
        max_iter=2000,
        class_weight='balanced'
    ))
])

# -------------------------------
# Decision Tree Model
# -------------------------------
tree_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('clf', DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced'
    ))
])

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Train Logistic Regression
# -------------------------------
print("Training Logistic Regression...")
log_model.fit(X_train, y_train)

# -------------------------------
# Tune & Train Decision Tree
# -------------------------------
print("Tuning Decision Tree...")
param_grid = {
    'clf__max_depth': [6, 8, 12, 16],
    'clf__min_samples_leaf': [5, 10, 20]
}

grid = GridSearchCV(
    tree_pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)
tree_model = grid.best_estimator_

# -------------------------------
# Predictions
# -------------------------------
log_pred = log_model.predict(X_test)
tree_pred = tree_model.predict(X_test)

log_prob = log_model.predict_proba(X_test)[:, 1]
tree_prob = tree_model.predict_proba(X_test)[:, 1]

# -------------------------------
# Metrics
# -------------------------------
print("\n--- MODEL PERFORMANCE ---")
print("Logistic Regression Accuracy:", round(accuracy_score(y_test, log_pred), 4))
print("Decision Tree Accuracy:", round(accuracy_score(y_test, tree_pred), 4))

print("\nLogistic ROC-AUC:", round(roc_auc_score(y_test, log_prob), 4))
print("Decision Tree ROC-AUC:", round(roc_auc_score(y_test, tree_prob), 4))

print("\nLogistic Classification Report:")
print(classification_report(y_test, log_pred))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, tree_pred))

# -------------------------------
# Confusion Matrix Graphs
# -------------------------------
ConfusionMatrixDisplay.from_predictions(
    y_test,
    log_pred,
    display_labels=["Rejected", "Approved"],
    cmap="Blues",
    values_format="d"
)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

ConfusionMatrixDisplay.from_predictions(
    y_test,
    tree_pred,
    display_labels=["Rejected", "Approved"],
    cmap="Greens",
    values_format="d"
)
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# -------------------------------
# Save models
# -------------------------------
joblib.dump(log_model, "logistic_model.joblib")
joblib.dump(tree_model, "decision_tree_model.joblib")

print("\nModels saved successfully.")
