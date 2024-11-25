import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Import necessary libraries for each model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


# For feature scaling
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Load your data
df = pd.read_csv("data/dataset2cleaned.csv")

# Split features and target label
X = df.drop(columns=['genre'])  # Features
y = df['genre']  # Target (genre label)


# # Random Forest ------------
# # Split into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the model
# rf_model = RandomForestClassifier(random_state=42)

# # Train the model
# rf_model.fit(X_train, y_train)

# # Make predictions
# y_pred = rf_model.predict(X_test)

# # Evaluate model
# print("RF Accuracy:", accuracy_score(y_test, y_pred))
# print("\nRF Classification Report:\n", classification_report(y_test, y_pred))


# # Decision Tree ------------

# # Split into training and testing sets
# X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the model
# dt_model = DecisionTreeClassifier(random_state=42)

# # Train the model
# dt_model.fit(X_train_dt, y_train_dt)

# # Make predictions
# y_pred_dt = dt_model.predict(X_test_dt)

# # Evaluate model
# print("Decision Tree Classifier")
# print("Accuracy:", accuracy_score(y_test_dt, y_pred_dt))
# print("\nClassification Report:\n", classification_report(y_test_dt, y_pred_dt))


# # SVM ------------
# # Feature Scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split into training and testing sets
# X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Initialize the model
# svm_model = SVC(random_state=42)

# # Train the model
# svm_model.fit(X_train_svm, y_train_svm)

# # Make predictions
# y_pred_svm = svm_model.predict(X_test_svm)

# # Evaluate model
# print("Support Vector Machine")
# print("Accuracy:", accuracy_score(y_test_svm, y_pred_svm))
# print("\nClassification Report:\n", classification_report(y_test_svm, y_pred_svm))

# # Logistic regression ------------
# # Feature Scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split into training and testing sets
# X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Initialize the model
# lr_model = LogisticRegression(random_state=42, max_iter=1000)

# # Train the model
# lr_model.fit(X_train_lr, y_train_lr)

# # Make predictions
# y_pred_lr = lr_model.predict(X_test_lr)

# # Evaluate model
# print("Logistic Regression")
# print("Accuracy:", accuracy_score(y_test_lr, y_pred_lr))
# print("\nClassification Report:\n", classification_report(y_test_lr, y_pred_lr))

# XGBoost ------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Split into training and testing sets
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize the model
xgb_model = xgb.XGBClassifier(random_state=42)

# Train the model
xgb_model.fit(X_train_xgb, y_train_xgb)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test_xgb)

# Evaluate model
print("XGBoost")
print("Accuracy:", accuracy_score(y_test_xgb, y_pred_xgb))
print("\nClassification Report:\n", classification_report(y_test_xgb, y_pred_xgb))



