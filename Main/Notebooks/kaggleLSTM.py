# from https://www.kaggle.com/shyambhu/sentiment-classification-using-lstm#
import numpy as np # linear algebra
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #ignore TensorFlow warnings
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Embedding
from keras.layers import Bidirectional
from nltk.corpus import stopwords
import re

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv('sentiment-analysis-on-movie-reviews/train.tsv.zip',sep = '\t')
test_data = pd.read_csv('sentiment-analysis-on-movie-reviews/train.tsv.zip',sep = '\t')
train_data.head()

train_data = train_data.drop(['PhraseId','SentenceId'],axis = 1)
test_data = test_data.drop(['PhraseId','SentenceId'],axis = 1)

max_features = 20000  # Only consider the top 20k words
maxlen = 200
train_data.head()

def text_cleaning(text):
    forbidden_words = set(stopwords.words('english'))
    if text:
        text = ' '.join(text.split('.'))
        text = re.sub('\/',' ',text)
        text = re.sub(r'\\',' ',text)
        text = re.sub(r'((http)\S+)','',text)
        text = re.sub(r'\s+', ' ', re.sub('[^A-Za-z]', ' ', text.strip().lower())).strip()
        text = re.sub(r'\W+', ' ', text.strip().lower()).strip()
        text = [word for word in text.split() if word not in forbidden_words]
        return text
    return []

train_data['flag'] = 'TRAIN'
test_data['flag'] = 'TEST'
total_docs = pd.concat([train_data,test_data],axis = 0,ignore_index = True)
total_docs['Phrase'] = total_docs['Phrase'].apply(lambda x: ' '.join(text_cleaning(x)))
phrases = total_docs['Phrase'].tolist()
from keras.preprocessing.text import one_hot
vocab_size = 50000
encoded_phrases = [one_hot(d, vocab_size) for d in phrases]
total_docs['Phrase'] = encoded_phrases
train_data = total_docs[total_docs['flag'] == 'TRAIN']
test_data = total_docs[total_docs['flag'] == 'TEST']
x_train = train_data['Phrase']
y_train = train_data['Sentiment']
x_val = test_data['Phrase']
y_val = test_data['Sentiment']

y_train.unique()

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

model = Sequential()
inputs = keras.Input(shape=(None,), dtype="int32")
# Embed each integer in a 128-dimensional vector
model.add(inputs)
model.add(Embedding(50000, 128))
# Add 2 bidirectional LSTMs
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
# Add a classifier
model.add(Dense(5, activation="sigmoid"))
#model = keras.Model(inputs, outputs)
model.summary()

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_val, y_val))

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val))