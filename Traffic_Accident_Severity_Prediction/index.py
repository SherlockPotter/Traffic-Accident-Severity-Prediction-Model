# ===============================
# Traffic Accident Severity Prediction
# ===============================

# Install required libraries
# !pip install pandas scikit-learn matplotlib seaborn

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# 2. Load dataset (from same folder as index.py)
DATASET_FILE = "dataset_traffic_accident_prediction1.csv"

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f"Dataset file '{DATASET_FILE}' not found in current folder.")

df = pd.read_csv(DATASET_FILE)

# 3. Quick Data Overview
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Preprocessing
# Drop duplicates
df = df.drop_duplicates()

# Handle missing values (forward fill for simplicity)
df = df.fillna(method='ffill')

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 5. Define features and target
target_column = "Accident_Severity"  # Change if your target column name is different
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset columns: {df.columns.tolist()}")

X = df.drop(columns=[target_column])
y = df[target_column]

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Predictions & Evaluation
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 9. Feature Importance
feature_importances = model.feature_importances_
features = df.drop(columns=[target_column]).columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
