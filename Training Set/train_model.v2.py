"""
train_model.py
--------------
Run this ONCE to train the model on the dataset and save it to disk.
The saved files (model.pkl, scaler.pkl, feature_columns.pkl) are then
loaded by app.py at startup — users never interact with this script.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH   = 'student_startup_success_dataset.csv'
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'
COLUMNS_PATH = 'feature_columns.pkl'

# ========== Data Preprocessing ==========
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")
print(df['success_label'].value_counts())

# ========== Feature Engineering ==========
df = df.drop(columns=['project_id', 'institution_name'])

df['ecosystem_support'] = df['mentorship_support'] + df['incubation_support']
df = df.drop(columns=['mentorship_support', 'incubation_support'])

df['startup_age'] = 2026 - df['year']
df = df.drop(columns=['year'])

df['funding_amount_usd_log'] = np.log1p(df['funding_amount_usd'])
df = df.drop(columns=['funding_amount_usd'])

# ========== Encoding ==========
df = pd.get_dummies(df, columns=['institution_type', 'project_domain'], drop_first=True)

# ========== Split ==========
X = df.drop('success_label', axis=1)
y = df['success_label']

print("\nClass distribution:\n", y.value_counts(normalize=True))
print("\nFeatures used:", list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ========== Scaling ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ========== Training ==========
# C=0.01 applies stronger regularization to prevent overconfident probabilities.
# The default C=1.0 produces extreme 0%/100% outputs on this dataset
# because the features are highly separable. C=0.01 keeps 92.7% accuracy
# while producing more realistic probability spread.
model = LogisticRegression(C=0.01, max_iter=1000)
model.fit(X_train_scaled, y_train)

# ========== Evaluation ==========
y_pred = model.predict(X_test_scaled)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ========== Save to Disk ==========
joblib.dump(model,            MODEL_PATH)
joblib.dump(scaler,           SCALER_PATH)
joblib.dump(list(X.columns),  COLUMNS_PATH)

print(f"\nModel saved to:   {MODEL_PATH}")
print(f"Scaler saved to:  {SCALER_PATH}")
print(f"Columns saved to: {COLUMNS_PATH}")
print("\nTraining complete. You can now run app.py.")
