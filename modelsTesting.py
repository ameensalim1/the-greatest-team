import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load your data
df = pd.read_csv("data/dataset2cleaned.csv")

# Split features and target label
X = df.drop(columns=['genre'])  # Features
y = df['genre']  # Target (genre label)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

example_song = np.array([[-0.7228298092513904,-0.3359452646913266,-0.17586311452816475,-1.6218922639370486,-1.1519222748982927,0.5127063798619507,-0.6851045147657994,-1.6196260080021874,-0.3142411391624565,1.0565395904525685,-0.46350689596578104,-0.35053548716678207,-1.5661477893408866,0.6404409905407521,-1.5858456445507447]])
predicted_genre = rf_model.predict(example_song)
print("Predicted Genre:", predicted_genre[0])