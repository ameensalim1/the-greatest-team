import keras
from keras import Sequential, layers, losses, regularizers
import keras_tuner
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# Model is exported to neural_network/best_model.keras


df = pd.read_csv("data/dataset2cleaned.csv")
x = df.drop(columns=['genre']).values
y = df['genre'].values 

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Split training data into validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

def createModel(hp):
  model = Sequential()
  model.add(layers.Input(shape=(15,)))
  
  # Hidden layers
  for i in range(hp.Int('num_layers', 3, 6)):
    model.add(
      layers.Dense(
        units=hp.Choice('units', [32, 64, 128]), 
        activation=hp.Choice('activation', ['relu', 'tanh', 'leaky_relu'])
      )
    )
    
  # Output layers
  model.add(
    layers.Dense(
      units=5, 
      activation=hp.Choice('out_activation', ['softmax', 'sigmoid'])
    )
  )
  
  model.compile(
    optimizer='adam',
    loss=losses.SparseCategoricalCrossentropy(), 
    metrics=['accuracy']
  )
  return model

# Grid search
tuner = keras_tuner.GridSearch(
  hypermodel=createModel,
  objective='val_accuracy',
  overwrite="true",
  directory="neural_network",
  project_name="history"
)

tuner.search(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

models = tuner.get_best_models()
best_model = models[0]
best_model.summary()

best_model.save('neural_network/best_model.keras')

y_pred = best_model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
print(accuracy_score(y_test, y_pred))