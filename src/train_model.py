import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load dataset
data_path = r"C:\Users\p\Downloads\archive (1)\telecom_churn.csv"
data = pd.read_csv(data_path)

print("Shape:", data.shape)
print(data.head())

# Check missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\nModel accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed report:\n", classification_report(y_test, y_pred))

# Save model
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/churn_model.joblib")
print("✅ Model saved as models/churn_model.joblib")
