import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier

# Load your data
df = pd.read_csv("data/dataset2cleaned.csv")

# Split features and target label
X = df.drop(columns=['genre']).values  # Features
y = df['genre'].values  # Target (genre label)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 1. Fine-Tuning Decision Tree
print("Tuning Decision Tree...")
param_grid_dt = {
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 3, 5, 10],
    'min_samples_leaf': [1, 2, 3, 5],
    'criterion': ['gini', 'entropy']
}
grid_search_dt = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid_dt,
    cv=10,
    scoring='accuracy',
    verbose=0,
    n_jobs=-1
)
grid_search_dt.fit(X_train, y_train)
best_dt = grid_search_dt.best_estimator_
y_pred_dt = best_dt.predict(X_test)
print("\nBest Parameters for Decision Tree:", grid_search_dt.best_params_)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))

# 2. Fine-Tuning SVM
print("\nTuning SVM...")
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
grid_search_svm = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=param_grid_svm,
    cv=5,
    scoring='accuracy',
    verbose=0,
    n_jobs=-1
)
grid_search_svm.fit(X_train, y_train)
best_svm = grid_search_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test)
print("\nBest Parameters for SVM:", grid_search_svm.best_params_)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# 3. Fine-Tuning Logistic Regression
print("\nTuning Logistic Regression...")
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear'],
    'penalty': ['l2'],  # Using L2 regularization
    'max_iter': [100, 200, 500]
}
grid_search_lr = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=param_grid_lr,
    cv=5,
    scoring='accuracy',
    verbose=0,
    n_jobs=-1
)
grid_search_lr.fit(X_train, y_train)
best_lr = grid_search_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test)
print("\nBest Parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))