import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap

import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import uniform, randint, mode
import tensorflow as tf
import tensorflow.python.keras as k
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import minmax_scale, MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
import xgboost as xgb
import catboost as cb
import eli5
from eli5.sklearn import PermutationImportance
from pprint import pprint
import librosa, IPython
from IPython.display import display
import librosa.display
from tqdm import tqdm
import random
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# set seed
seed = 42
random.seed = seed
np.random.seed(seed)
tf.random.set_seed(seed)

# load dataset
data = pd.read_csv('./Data/features_30_sec.csv')
print(data.head())


# about dataset
print("Dataset has",data.shape)
print("Count of Positive and Negative samples")z
data.label.value_counts().reset_index()


# visualize data
audio_fp = './Data/genres_original/blues/blues.00000.wav'
audio_data, sr = librosa.load(audio_fp)
audio_data, _ = librosa.effects.trim(audio_data)
plt.figure(figsize=(15,5))
librosa.display.waveshow(audio_data)
plt.show()

# default FFT window size
n_fft = 2048 # window size
hop_length = 512 # window hop length for STFT
stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
stft_db = librosa.amplitude_to_db(stft, ref=np.max)

plt.figure(figsize=(12,4))
librosa.display.specshow(stft, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title("Spectrogram with amplitude")
plt.show()

plt.figure(figsize=(12,4))
librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='log', cmap='cool')
plt.colorbar()
plt.title("Spectrogram with decibel log")
plt.show()

# plot Mel Spectrogram
mel_spec = librosa.feature.melspectrogram(audio_data, sr=sr)
mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
plt.figure(figsize=(16,6))
librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='cool')
plt.colorbar()
plt.title("Mel Spectrogram")
plt.show()

# plot PCA on Genres
data_ = data.iloc[0:, 1:]
y = data_['label']
X = data_.loc[:, data_.columns != 'label']

# normalize
cols = X.columns
min_max_scaler = MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled, columns = cols)

# Top 2 pca components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

# concatenate with target label
finalDf = pd.concat([principalDf, y], axis = 1)
 
plt.figure(figsize = (16, 9))
sns.scatterplot(x = "pc1", y = "pc2", data = finalDf, hue = "label", alpha = 0.7, s = 100);

plt.title('PCA on Genres', fontsize = 20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 10);
plt.xlabel("Principal Component 1", fontsize = 15)
plt.ylabel("Principal Component 2", fontsize = 15)

# encode label
label_encoder = LabelEncoder()
labels = data['label']
label_encoder.fit(labels)
data.insert(60, 'label_id', 9999)
data.insert(1, 'filename_full', '')
for i in range(len(data)):
    label = data.loc[i,'label']
    label_id =label_encoder.transform([label])
    data.loc[i,'label_id']=label_id.item()
    data.loc[i,'filename_full']=str(data.loc[i,'filename']).split('.')[0]+"."+str(data.loc[i,'filename']).split('.')[1]+"."+str(data.loc[i,'filename']).split('.')[3]
data['label_id']=data['label_id'].astype(int)

# data preprocess
X_full = data.drop(['filename','filename_full', 'length','label', 'label_id'], axis = 1)
y_full = data['label_id'].astype('int')
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state = seed, shuffle = True)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

# train and teat classic machine learning models
def train_and_predict(model):
    model.fit(X_train,y_train)
    y_pred_train = model.predict(X_train)
    print(classification_report(y_train, y_pred_train, digits=3))
    print('Train Accuracy = {:.4f}'.format(accuracy_score(y_train, y_pred_train)))
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_confusion_matrix(model, X_train, y_train, display_labels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"], cmap=plt.cm.Blues, xticks_rotation=90, ax=ax)
    plt.show()
    y_pred_test = model.predict(X_test)
    print(classification_report(y_test, y_pred_test, digits=3))
    print('Test Accuracy = {:.4f}'.format(accuracy_score(y_test, y_pred_test)))
    print("ROC_AUC Score: "+ str(multiclass_roc_auc_score(y_test,y_pred_test)))
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_confusion_matrix(model, X_test, y_test, display_labels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"], cmap=plt.cm.Blues, xticks_rotation=90, ax=ax)
    plt.show()
   

rfc = RandomForestClassifier(random_state=seed, n_jobs=-1)
train_and_predict(rfc)
