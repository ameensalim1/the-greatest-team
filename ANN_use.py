import keras
from keras import Sequential, layers, losses
import keras_tuner
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/dataset2cleaned.csv")

x = df.drop(columns=['genre']).values
y = df['genre'].values 

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model: Sequential = keras.models.load_model('neural_network/best_model.keras')

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

print("Accuracy: ", accuracy_score(y_test, y_pred))
