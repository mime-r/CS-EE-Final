# Silence Tensorflow
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import os
import pickle
import librosa
import librosa.display
from IPython.display import Audio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
# print(tf.__version__)
from tensorflow import keras as k



df = pd.read_csv('./Data/original_30.csv')

df.head()
df = df.drop(labels="filename", axis=1)

class_list = df.iloc[:, -1]

convertor = LabelEncoder()
y = convertor.fit_transform(class_list)

from sklearn.preprocessing import StandardScaler

fit = StandardScaler()
X = fit.fit_transform(np.array(df.iloc[:, :-1], dtype=float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train, X_test, y_train, y_test)


def trainModel(model, epochs, optimizer):
    batch_size = 128
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model.fit(X_train,
                     y_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_data=(X_test, y_test))


def plotValidate(history):
    print("Validation Accuracy, ", max(history.history['val_accuracy']))
    pd.DataFrame(history.history).plot(figsize=(12, 6))
    plt.show()


model = k.models.Sequential([
    k.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1], )),
    k.layers.Dropout(0.2),
    k.layers.Dense(256, activation='relu'),
    k.layers.Dropout(0.2),
    k.layers.Dense(128, activation='relu'),
    k.layers.Dropout(0.2),
    k.layers.Dense(64, activation='relu'),
    k.layers.Dropout(0.2),
    k.layers.Dense(10, activation='softmax'),
])
# print(model.summary())

print("*** Training Model ***")
model_history = trainModel(model=model, epochs=300, optimizer='adam')

print("*** Evaluating Model ***")
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=128)
# print("Test Loss: ", test_loss)
print("Accuracy:", test_acc * 100, "%")

plotValidate(model_history)
