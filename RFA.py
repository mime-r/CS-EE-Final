import os
import pandas as pd
import numpy as np
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

try:
    os.chdir("./Final/")
except:
    pass

# data = pd.read_csv('./Data/features_30_sec.csv')
data = pd.read_csv('./Data/features_30_sec.csv')
data = data.iloc[0:, 1:]    

y = data['label']
X = data.loc[:, data.columns != 'label']
cols = X.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled, columns = cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# print(X_train, X_test, y_train, y_test)

model = RandomForestClassifier(n_estimators=1000, max_depth=30, random_state=0)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, preds)*100, "%")