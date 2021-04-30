# DataFrame
import pandas as pd
# Matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import GRU, Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM, \
    SpatialDropout1D, Bidirectional, TimeDistributed
from keras.losses import binary_crossentropy
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Word2vec
import gensim

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools
import MethodHandler
import glob

import tensorflow as tf

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 15
BATCH_SIZE = 512

DATASET_ENCODING = "ISO-8859-1"

# SENTIMENT
POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "neutral"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# dataset paths

# dirPath = "C:\\Users\\Jonas\\PycharmProjects\\MLNLP\\Main\\Data\\manually_labelled"  # "  # Jonas path
# dirPath = "C:\\Users\\HE400\\PycharmProjects\\MLNLP_main\\Main\\Data\\manually_labelled"  # "  # Hammi path
dirPath = "C:\\Users\\mail\\PycharmProjects\\\MLNLP\\Main\\Data\\manually_labelled"  # Jonas path work
# savepath = "C:\\Users\\Jonas\\PycharmProjects\\MLNLP\\Main\\Code\\save"  # Jonas path
# savepath = "C:\\Users\\HE400\\PycharmProjects\\MLNLP_main\\Main\\Code\\save" # Hammi path
savepath = "C:\\Users\\mail\\PycharmProjects\\\MLNLP\\Main\\Code\\save\\manual_eval"  # Jonas path work

model_path = "C:\\Users\\mail\\PycharmProjects\\MLNLP\\Main\\Code\\save\\Run " \
             "23-04-21\\models\\*.h5"
tokenizer_path = "C:\\Users\\mail\\PycharmProjects\\MLNLP\\Main\\Code\\save\\Run " \
                 "23-04-21\\models\\*.pkl"
outside_path = "C:\\Users\\mail\\PycharmProjects\\MLNLP\\Main\\Code\\save\\"
inside_path = "\\models"
list_dates = ["Run 23-04-21", "Run 24-04-21", "Run 25-04-21 1", "Run 26-04-21"]
datasets = ["Airline_tweets", "Financial_news", "IMDB", "Reddit", "Steam", "Twitter"]
list_paths_model = []
list_paths_tokenizer = []


for j in (list_dates):
    path = os.path.join(outside_path + j + inside_path)
    path_model = os.path.join(path + '\\*.h5')
    path_tokenizer = os.path.join(path + '\\*.pkl')
    list_paths_model.append(path_model)
    list_paths_tokenizer.append(path_tokenizer)

    # for f in range(len(list_paths_model)):
    #     list1 = glob.glob(list_paths_model[f])
    #     list2 = glob.glob(list_paths_tokenizer[f])

    list1 = glob.glob(path_model)
    list2 = glob.glob(path_tokenizer)

    for i in range(len(list1)):
        print(model_path[i])
        model_path = list1[i]
        tokenizer_path = list2[i]
        model = MethodHandler.loadModel(model_path)
        tokenizer = MethodHandler.loadTokenizer(tokenizer_path)
        print(model_path)
        print(tokenizer_path)
        print(model.name)
        dataset_trained_on = datasets[i]

        model_name = model.name + date + dataset_trained_on

        ts = time.gmtime()
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", ts)

        filename = "Scraped_merged_manually_labelled.csv"
        file = os.path.join(dirPath, filename)

        DATASET_COLUMNS = ["text", "Ticks", "target", "Score", "Date", "URL"]
        df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, skiprows=1)
        print(df['target'].unique())
        pred = []
        predScore = []


        if os.path.isfile(file):
            print(file)
        print("Open file:", dirPath + "\\" + filename)
        print("Dataset size:", len(df))

        dfVal, dfExtVal = train_test_split(df, test_size= 0.5, random_state=42)
        print("Validation size:", len(dfVal))
        print("External Evaluation size:", len(dfExtVal))

        y_val = dfVal.target
        y_extVal = dfExtVal.target

        x_val = dfVal.text
        x_val_padded = pad_sequences(tokenizer.texts_to_sequences(dfVal.text), maxlen=SEQUENCE_LENGTH)
        x_extVal = pad_sequences(tokenizer.texts_to_sequences(dfExtVal.text), maxlen=SEQUENCE_LENGTH)



        def plot_confusion_matrix(cm, classes,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """

            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title, fontsize=30)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=90, fontsize=22)
            plt.yticks(tick_marks, classes, fontsize=22)

            fmt = '.2f'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label', fontsize=25)
            plt.xlabel('Predicted label', fontsize=25)


        def decode_sentiment(score, include_neutral=True):
            if include_neutral:
                label = NEUTRAL
                if score <= SENTIMENT_THRESHOLDS[0]:
                    label = 0
                elif score >= SENTIMENT_THRESHOLDS[1]:
                    label = 1

                return label
            else:
                return NEGATIVE if score < 0.5 else POSITIVE

        # score = model.evaluate(x_val_padded, y_val, batch_size=BATCH_SIZE)
        # print("ACCURACY:", score[1])
        # print("LOSS:", score[0])

        counter = len(x_val)
        for i in x_val:
            score = MethodHandler.predictSentiment(i, model=model, tokenizer=tokenizer)
            pred.append(decode_sentiment(score, include_neutral=False))
            predScore.append(score)
            print(counter)
            counter = counter-1

        cm = confusion_matrix(y_val, pred)
        print(cm)
        plt.figure(figsize=(12, 12))
        plot_confusion_matrix(cm, dfVal.target.unique(), title="Confusion matrix")
        fig = plt.gcf()
        plt.show()

        os.chdir(savepath)
        fig.savefig(model_name + "cm_val_" + filename + '.png')

        print(classification_report(y_val, pred))
        score_acc = accuracy_score(y_val, pred)
        print("printing acc score: ", score_acc)

        report_val = classification_report(y_val, pred, output_dict=True)

        reportData = pd.DataFrame(report_val).transpose()
        reportDataName = model_name + "_classificationReport_val_" + ".csv"
        reportData.to_csv(reportDataName, index=False)




        # def validation_metrics(model, x, dfv):
        #     y_test = list(dfv.target)
        #     predictions = model.predict(x)
        #     y_pred = [decode_sentiment(score, include_neutral=False) for score in predictions]
        #
        #     cnf_matrix_test = confusion_matrix(y_test, y_pred, labels=["positive", "negative"])
        #
        #     plt.figure(figsize=(12, 12))
        #     plot_confusion_matrix(cnf_matrix_test, classes=dfv.target.unique(), title="Confusion matrix")
        #     fig_cm_test = plt.gcf()
        #     plt.show()
        #
        #     fig_cm_test.savefig(model.name + "cm_test_" + filename + ts + '.png')
        #
        #     report_test = classification_report(y_test, y_pred, output_dict=True)
        #
        #     reportData = pd.DataFrame(report_test).transpose()
        #     reportDataName = model.name + "_classificationReport_test_" + filename + ts + ".csv"
        #     reportData.to_csv(reportDataName, index=False)


        #validation_metrics(model, x_val, dfVal)

