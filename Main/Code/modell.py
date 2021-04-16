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
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
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

import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Initializing paramteters:

# TEXT CLEANING regex
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 1
BATCH_SIZE = 512

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# DATASET
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

# dataset paths

dirPath = "C:\\Users\\Jonas\\PycharmProjects\\MLNLP\\Main\\Data\\Labelled"


def amazing():
    def decode_sentiment(label):
        return decode_map[int(label)]

    # This is where the decode_sentiment method looks through the target column and switches numbers to strings (negative,positive)

    # print("DATATYPE: ", type(df.target[1]))

    # df.target = df.target.apply(lambda x: decode_sentiment(x))
    # df.target is of class type "numpy.int64" => Integer (-9223372036854775808 to 9223372036854775807) Lambda function converts this..


    for filename in os.listdir(dirPath):

        file = os.path.join(dirPath, filename)
        # df = pd.read_csv(file, encoding=DATASET_ENCODING)
        if filename == "Airline_Tweets.csv":
            # decode_map = {"negative": "NEGATIVE", "neutral": "NEUTRAL", "positive": "POSITIVE"}
            DATASET_COLUMNS = ["tweet_id", "target", "airline_sentiment_confidence", "negativereason",
                               "negativereason_confidence", "airline", "airline_sentiment_gold", "name", "negativereason_gold", "retweet_count",
                               "text", "tweet_coord", "tweet_created", "tweet_location", "user_timezone"]
            df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, skiprows=1)
            #df.target = df.target
            print("Printing DF", df.target)
        elif filename == "Financial_news_all-data.csv":
            DATASET_COLUMNS = ["target", "text"]
            df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
        elif filename == "IMDB_Dataset.csv":
            DATASET_COLUMNS = ["target", "text"]
            df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, skiprows=1)
        elif filename == "Reddit_data.csv":
            decode_map = {-1: "NEGATIVE", 0: "NEUTRAL", 1: "POSITIVE"}
            DATASET_COLUMNS = ["text", "target"]
            df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, skiprows=1)
            df.target = df.target.apply(lambda x: decode_sentiment(x))
        elif filename == "Steam_train.csv":
            DATASET_COLUMNS = ["review_id", "title", "year", "text", "target"]
            decode_map = {1: "NEGATIVE",  0: "POSITIVE"}
            df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, skiprows=1)
            df.target = df.target.apply(lambda x: decode_sentiment(x))
        elif filename == "Twitter_data.csv":
            DATASET_COLUMNS = ["text", "target"]
            decode_map = {-1: "NEGATIVE", 0: "NEUTRAL", 1: "POSITIVE"}
            df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, skiprows=1)
            df.target = df.target.apply(lambda x: decode_sentiment(x))
        #elif filename == "sentiment140.csv":
        #    decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
        #    DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
        #    df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
        #    df.target = df.target.apply(lambda x: decode_sentiment(x))

        # checking if it is a file
        if os.path.isfile(file):
            print(file)
        print("Open file:", dirPath + filename)
        print("Dataset size:", len(df))

        def amazing2():

            # DATASET

            # EXPORT
            # KERAS_MODEL = "model.h5"
            # WORD2VEC_MODEL = "model.w2v"
            # TOKENIZER_MODEL = "tokenizer.pkl"
            # ENCODER_MODEL = "encoder.pkl"
            # Print 5 rows:
            # df.head(5)

            # Used in decode_sentiment for 'labeling data?'

            stop_words = stopwords.words("english")
            stemmer = SnowballStemmer("english")

            def preprocess(text, stem=False):  # false or true determines whether the method uses stemmer for tokens.
                # Remove link,user and special characters
                text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
                tokens = []
                for token in text.split():
                    if token not in stop_words:
                        if stem:
                            tokens.append(stemmer.stem(token))
                        else:
                            tokens.append(token)
                return " ".join(tokens)  # String with all tokens seperated with ' '

            # Updates the text column in the dataset and cleans it using the preprocess method
            df.text = df.text.apply(
                lambda x: preprocess(x))

            dfTrain, dfTest = train_test_split(df, test_size=1 - 0.8, random_state=42)
            print("TRAINING size:", len(dfTrain))
            print("TESTING size:", len(dfTest))

            dfTrain, dfVal = train_test_split(dfTrain, test_size=0.25,
                                              random_state=1)  # 0.25 x 0.8 = 0.2  #RANDOM STATE NEEDS ATTENTION
            print('Train size updated: ', len(dfTrain))
            print('eval test size: ', len(dfVal))

            docs = []
            for t in dfTrain.text:
                docs.append(t.split())

            # ***********************************************************************************************************************

            # W2V model initialized
            w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE,
                                                        window=W2V_WINDOW,
                                                        min_count=W2V_MIN_COUNT,
                                                        workers=8)
            w2v_model.build_vocab(docs)  # prepare the model vocabulary

            words = w2v_model.wv.vocab.keys()  # Access the words in vocabulary with .keys()
            vocab_size = len(words)
            print("Vocab size", vocab_size)  # Print length of vocabulary

            w2v_model.train(docs, total_examples=len(docs), epochs=W2V_EPOCH)
            # ***********************************************************************************************************************
            # Initialize tokenizer
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(dfTrain.text)
            vocab_size = len(tokenizer.word_index) + 1
            print("Total words", vocab_size)

            x_train = pad_sequences(tokenizer.texts_to_sequences(dfTrain.text), maxlen=SEQUENCE_LENGTH)
            x_test = pad_sequences(tokenizer.texts_to_sequences(dfTest.text), maxlen=SEQUENCE_LENGTH)
            x_val = pad_sequences(tokenizer.texts_to_sequences(dfVal.text), maxlen=SEQUENCE_LENGTH)  # TEST TO EVAL?

            labels = dfTrain.target.unique().tolist()
            print(labels)

            encoder = LabelEncoder()
            encoder.fit(dfTrain.target.tolist())

            y_train = encoder.transform(dfTrain.target.tolist())
            y_test = encoder.transform(dfTest.target.tolist())
            y_val = encoder.transform(dfVal.target.tolist())  # TEST TO EVAL?

            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)
            print("x_train", x_train.shape)
            print("y_train", y_train.shape)
            print()
            print("x_test", x_test.shape)
            print("y_test", y_test.shape)

            print(y_train[:10])

            embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
            for word, i in tokenizer.word_index.items():
                if word in w2v_model.wv:
                    embedding_matrix[i] = w2v_model.wv[word]

            embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH,
                                        trainable=False)

            # Initialize sequential model (keras)
            modelList = []

            model = Sequential(name='model')
            model.add(embedding_layer)
            model.add(Dropout(0.5))
            model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(1, activation='sigmoid'))

            model.compile(loss='binary_crossentropy',
                          optimizer="adam",
                          metrics=['accuracy'])
            modelList.append(model)
            # Model_1
            model_1 = Sequential(name='model_1')
            model_1.add(embedding_layer)
            model_1.add(Dropout(0.5))
            model_1.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
            model_1.add(Dense(1, activation='relu'))


            model_1.compile(loss='binary_crossentropy',
                            optimizer="adam",
                            metrics=['accuracy'])
            modelList.append(model_1)

            # not sure what dis does
            callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),  # Needs to be looked into****ØØØØØ
                         EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]

            def trainAndEval(modelName):
                modelName.summary()
                history = modelName.fit(x_train, y_train,
                                        batch_size=BATCH_SIZE,
                                        epochs=EPOCHS,
                                        validation_split=0.1,
                                        verbose=1,
                                        callbacks=callbacks)
                # EVALUATE WITH THE EVALUATION SET:
                score = modelName.evaluate(x_val, y_val, batch_size=BATCH_SIZE)
                print(modelName.name, "done training - ...")
                return history, score

            for model in modelList:
                history, score = trainAndEval(model)
                MethodHandler.saveModel(model, tokenizer, filename)
                # history = model.fit(x_train, y_train,
                #                     batch_size=BATCH_SIZE,
                #                     epochs=EPOCHS,
                #                     validation_split=0.1, #???????????????
                #                     verbose=1,
                #                     callbacks=callbacks)

                # score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
                print()
                print("ACCURACY:", score[1])
                print("LOSS:", score[0])

                # Data Visualization for eval
                acc = history.history['accuracy']
                val_acc = history.history['val_accuracy']
                loss = history.history['loss']
                val_loss = history.history['val_loss']

                epochs = range(len(acc))

                plt.plot(epochs, acc, 'b', label='Training acc')
                plt.plot(epochs, val_acc, 'r', label='Validation acc')
                plt.title('Training and validation accuracy')
                plt.legend()

                plt.figure()

                plt.plot(epochs, loss, 'b', label='Training loss')
                plt.plot(epochs, val_loss, 'r', label='Validation loss')
                plt.title('Training and validation loss')
                plt.legend()

                plt.show()

            # SENTIMENT DECODE METHOD

            def decode_sentiment(score, include_neutral=True):
                if include_neutral:
                    label = NEUTRAL
                    if score <= SENTIMENT_THRESHOLDS[0]:
                        label = NEGATIVE
                    elif score >= SENTIMENT_THRESHOLDS[1]:
                        label = POSITIVE

                    return label
                else:
                    return NEGATIVE if score < 0.5 else POSITIVE

            # EVAL PARAMS FOR CONFUSION MATRIX
            y_pred_1d = []
            y_test_1d = list(dfTrain.target)
            predictions = model.predict(x_test, verbose=1, batch_size=8000)
            y_pred_1d = [decode_sentiment(score, include_neutral=False) for score in predictions]

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

            return model, model_1, tokenizer
        amazing2()


# cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
# plt.figure(figsize=(12, 12))
# plot_confusion_matrix(cnf_matrix, classes=dfTrain.target.unique(), title="Confusion matrix")
# plt.show()
# print(classification_report(y_test_1d, y_pred_1d))
# accuracy_score(y_test_1d, y_pred_1d)
