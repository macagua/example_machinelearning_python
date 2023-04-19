""" Make Your First AI in 15 Minutes with Python

Source:
    https://www.youtube.com/watch?v=z1PGJ9quPV8
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Read the dataset
dataset = pd.read_csv('data/cancer.csv')

# Set x axis
x = dataset.drop(columns=["diagnostics(1=m, 0=b)"])

# Set x axis
y = dataset["diagnostics(1=m, 0=b)"]

# Split arrays or matrices into random train and test subsets for x and y axes.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Define Sequential model
model = tf.keras.models.Sequential()

# Add 3 layers to Sequential model
model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape, activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
print("Compile the model")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Train the model")
model.fit(x_train, y_train, epochs=1000)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
model.evaluate(x_test, y_test)
