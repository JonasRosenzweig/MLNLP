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

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
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
POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "neutral"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# DATASET
# DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

# dataset paths

# dirPath = "C:\\Users\\Jonas\\PycharmProjects\\MLNLP\\Main\\Data\\Labelled"  # Jonas path
#dirPath = "C:\\Users\\HE400\\PycharmProjects\\MLNLP_main\\Main\\Data\\Labelled"  # Hammi path
dirPath = "C:\\Users\\mail\\PycharmProjects\\\MLNLP\\Main\\Data\\Labelled"  # Jonas path work
# savepath = "C:\\Users\\Jonas\\PycharmProjects\\MLNLP\\Main\\Code\\save"  # Jonas path
#savepath = "C:\\Users\\HE400\\PycharmProjects\\MLNLP_main\\Main\\Code\\save"  # Hammi path
savepath = "C:\\Users\\mail\\PycharmProjects\\\MLNLP\\Main\\Code\\save"  # Jonas path work

ts = time.gmtime()
ts = time.strftime("%Y-%m-%d_%H-%M-%S", ts)


def amazing():
    def decode_sentiment(label):
        return decode_map[int(label)]

    def decode_sentiment2(label):
        return decode_map[str(label)]

    def diff(positive, negative):
        if len(positive) > len(negative):
            difference = len(positive) - len(negative)
            return len(positive) - difference

        elif len(negative) > len(positive):
            difference = len(negative) - len(positive)
            return len(negative) - difference

        else:
            print('something went wrong..', len(positive), len(negative))

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
                               "negativereason_confidence", "airline", "airline_sentiment_gold", "name",
                               "negativereason_gold", "retweet_count",
                               "text", "tweet_coord", "tweet_created", "tweet_location", "user_timezone"]
            df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, skiprows=1)
            df = df[df.target != "neutral"]  # removes neutral

            positive = df[df['target'] == "positive"]
            negative = df[df['target'] == "negative"]

            if len(positive) != len(negative):
                positive = positive[:diff(positive, negative)]
                negative = negative[:diff(positive, negative)]
                df = pd.concat([positive, negative])

        # elif filename == "sentiment140.csv":
        #     DATASET_COLUMNS = ["target", "id", "date", "flag", "user", "text"]
        #     decode_map = {0: "negative", 4: "positive"}
        #     df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, skiprows=1)
        #     df.target = df.target.apply(lambda x: decode_sentiment(x))
        #     df = df[df.target != "neutral"]
        #
        #     positive = df[df['target'] == "positive"]
        #     negative = df[df['target'] == "negative"]
        #     if len(positive) != len(negative):
        #         positive = positive[:diff(positive, negative)]
        #         negative = negative[:diff(positive, negative)]
        #         df = pd.concat([positive, negative])

        # elif filename == "Scraped_merged_manually_labelled.csv":  # `?????????????
        #     DATASET_COLUMNS = ["text", "Ticks", "target", "Score", "Date", "URL"]
        #     df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, skiprows=1)
        #     positive = df[df['target'] == "positive"]
        #     negative = df[df['target'] == "negative"]
        #     if len(positive) != len(negative):
        #         positive = positive[:diff(positive, negative)]
        #         negative = negative[:diff(positive, negative)]
        #         df = pd.concat([positive, negative])

        # elif filename == "Financial_news_all-data.csv":
        #     DATASET_COLUMNS = ["target", "text"]
        #     df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
        #     df = df[df.target != "neutral"]  # removes neutral
        #
        #     positive = df[df['target'] == "positive"]
        #     negative = df[df['target'] == "negative"]
        #
        #     if len(positive) != len(negative):
        #         positive = positive[:diff(positive, negative)]
        #         negative = negative[:diff(positive, negative)]
        #         df = pd.concat([positive, negative])

        elif filename == "IMDB_Dataset.csv":
            DATASET_COLUMNS = ["text", "target"]
            df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, skiprows=1)
            df = df[df.target != "neutral"]  # removes neutral
            decode_map = {1: POSITIVE, 0: NEGATIVE}

            positive = df[df['target'] == "positive"]
            negative = df[df['target'] == "negative"]

            if len(positive) != len(negative):
                positive = positive[:diff(positive, negative)]
                negative = negative[:diff(positive, negative)]
                df = pd.concat([positive, negative])


        elif filename == "Reddit_data.csv":
            decode_map = {-1: "negative", 1: "positive"}
            DATASET_COLUMNS = ["text", "target"]
            df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, skiprows=1)
            df.target = df.target.apply(lambda x: decode_sentiment(x))
            df = df[df.target != "neutral"]  # removes neutral

            positive = df[df['target'] == "positive"]
            negative = df[df['target'] == "negative"]

            if len(positive) != len(negative):
                positive = positive[:diff(positive, negative)]
                negative = negative[:diff(positive, negative)]
                df = pd.concat([positive, negative])



        elif filename == "Steam_train.csv":
            DATASET_COLUMNS = ["review_id", "title", "year", "text", "target"]
            decode_map = {1: "negative", 0: "positive"}
            df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, skiprows=1)
            df.target = df.target.apply(lambda x: decode_sentiment(x))
            df = df[df.target != "neutral"]  # removes neutral

            positive = df[df['target'] == "positive"]
            negative = df[df['target'] == "negative"]

            if len(positive) != len(negative):
                positive = positive[:diff(positive, negative)]
                negative = negative[:diff(positive, negative)]
                df = pd.concat([positive, negative])

        # elif filename == "Twitter_data.csv":
        #     DATASET_COLUMNS = ["text", "target"]
        #     decode_map = {-1: "negative", 1: "positive"}
        #     df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, skiprows=1)
        #     df.target = df.target.apply(lambda x: decode_sentiment(x))
        #     df = df[df.target != "neutral"]  # removes neutral
        #
        #     positive = df[df['target'] == "positive"]
        #     negative = df[df['target'] == "negative"]
        #
        #     if len(positive) != len(negative):
        #         positive = positive[:diff(positive, negative)]
        #         negative = negative[:diff(positive, negative)]
        #         df = pd.concat([positive, negative])

        # elif filename == "sentiment140.csv":
        #    decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
        #    DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
        #    df = pd.read_csv(file, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
        #    df.target = df.target.apply(lambda x: decode_sentiment(x))

        # checking if it is a file
        if os.path.isfile(file):
            print(file)
        print("Open file:", dirPath + "\\" + filename)
        print("Dataset size:", len(df))

        # target_cnt = Counter(df.target)

        # plt.figure(figsize=(16, 8))
        # plt.bar(target_cnt.keys(), target_cnt.values())
        # plt.title("Dataset labels Distribution")

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

            docs = []
            for t in df.text:  # dfTrain
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
            tokenizer.fit_on_texts(df.text)  # dfTrain
            vocab_size = len(tokenizer.word_index) + 1
            print("Total words", vocab_size)

            x_train = pad_sequences(tokenizer.texts_to_sequences(df.text), maxlen=SEQUENCE_LENGTH)  # dfTrain

            labels = df.target.unique().tolist()  # dfTrain
            print(labels)

            encoder = LabelEncoder()
            encoder.fit(df.target.tolist())  # dfTrain

            y_train = encoder.transform(df.target.tolist())  # dfTrain

            y_train = y_train.reshape(-1, 1)

            print('YTRAIN:', y_train)

            embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
            for word, i in tokenizer.word_index.items():
                if word in w2v_model.wv:
                    embedding_matrix[i] = w2v_model.wv[word]

            embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH,
                                        trainable=False)

            # Initialize sequential model (keras)
            modelList = []

            model_0 = Sequential(name='OG_LSTM')
            model_0.add(embedding_layer)
            model_0.add(Dropout(0.5))
            model_0.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
            model_0.add(Dense(1, activation='sigmoid'))

            model_0.compile(loss='binary_crossentropy',
                            optimizer="adam",
                            metrics=['accuracy'])
            modelList.append(model_0)
            #
            #
            # model_1 = Sequential(name='model_1')
            # model_1.add(embedding_layer)
            # model_1.add(Dropout(0.5))
            # model_1.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
            # model_1.add(Dense(1, activation='relu'))
            #
            # model_1.compile(loss='binary_crossentropy',
            #                 optimizer="adam",
            #                 metrics=['accuracy'])
            # modelList.append(model_1)
            #
            # model_2 = Sequential(name='model_2')
            # model_2.add(embedding_layer)
            # model_2.add(SpatialDropout1D(0.7))
            # model_2.add(LSTM(64, dropout=0.7, recurrent_dropout=0.7))
            # model_2.add(Dense(1, activation='softmax'))
            #
            # model_2.compile(optimizer='adam',
            #                 loss='categorical_crossentropy',
            #                 metrics=['accuracy'])
            # modelList.append(model_2)

            # model_3 = Sequential(name='GRU_1')
            # model_3.add(embedding_layer)
            # model_3.add(Dropout(0.5))
            # model_3.add(GRU(100))
            # model_3.add(Dense(1, activation='sigmoid'))
            # model_3.compile(loss='binary_crossentropy',
            #                 optimizer='adam',
            #                 metrics=['accuracy'])
            # modelList.append(model_3)

            # model_3 = Sequential(name='LSTM_Bidirectional_1')
            # model_3.add(embedding_layer)
            # model_3.add(Dropout(0.5))
            # model_3.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
            # model_3.add(Dense(1, activation='sigmoid'))
            #
            # model_3.compile(loss='binary_crossentropy',
            #                 optimizer="adam",
            #                 metrics=['accuracy'])
            # modelList.append(model_3)
            #
            # model_4 = Sequential(name='GRU_1')
            # model_4.add(embedding_layer)
            # model_4.add(Dropout(0.5))
            # model_4.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
            # model_4.add(Dense(1, activation='sigmoid'))
            # model_4.compile(loss='binary_crossentropy',
            #                 optimizer='adam',
            #                 metrics=['accuracy'])
            # modelList.append(model_4)

            # model_5 = Sequential(name='LSTM_Bidirectional_1')
            # model_5.add(embedding_layer)
            # model_5.add(Bidirectional(LSTM(100)))
            # model_5.add(TimeDistributed(Dense(1, activation='sigmoid')))
            # model_5.compile(loss='binary_crossentropy',
            #                 optimizer='adam',
            #                 metrics=['accuracy'])
            # modelList.append(model_5)

            # not sure what dis does
            callbacks = [EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=5)]

            # EarlyStopping(monitor='val_loss', min_delta=0.0001)
            # ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0)

            # evaldf = pd.read_csv(
            #     'C:\\Users\\HE400\\PycharmProjects\\MLNLP_main\\Main\\Data\\manually_labelled\\Scraped_merged_manually_labelled.csv',
            #     names=['Comments', 'Ticks', 'Sentiment', 'Score', 'Date', 'Url'], skiprows=1)

            evaldf = pd.read_csv(
                'C:\\Users\\mail\\PycharmProjects\\MLNLP\\Main\\Data\\manually_labelled\\Scraped_merged_manually_labelled.csv',
                names=['Comments', 'Ticks', 'Sentiment', 'Score', 'Date', 'Url'], skiprows=1)

            evaldf_val = evaldf[:int((len(evaldf) * 0.4))]
            evaldf_test = evaldf[int((len(evaldf) * 0.6)):]

            encoder.fit(evaldf_val.Sentiment.tolist())
            y_val = encoder.transform(evaldf_val.Sentiment.tolist())
            y_val = y_val.reshape(-1, 1)
            x_val = pad_sequences(tokenizer.texts_to_sequences(evaldf_val.Comments), maxlen=SEQUENCE_LENGTH)

            encoder.fit(evaldf_val.Sentiment.tolist())
            y_test = encoder.transform(evaldf_test.Sentiment.tolist())
            y_test = y_test.reshape(-1, 1)
            x_test = pad_sequences(tokenizer.texts_to_sequences(evaldf_test.Comments), maxlen=SEQUENCE_LENGTH)

            def trainAndEval(modelName):
                modelName.summary()
                history = modelName.fit(x_train, y_train,
                                        batch_size=BATCH_SIZE,
                                        epochs=EPOCHS,
                                        validation_split=0.1,
                                        verbose=1,
                                        callbacks=callbacks)

                # EVALUATE WITH THE EVALUATION SET: (internal eval)
                score = modelName.evaluate(x_val, y_val, batch_size=BATCH_SIZE)
                print(modelName.name, "done training - ...")
                return history, score

            for model in modelList:

                history, score = trainAndEval(model)
                MethodHandler.saveModel(model, tokenizer, filename)

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

                save_path = savepath
                os.chdir(save_path)

                fig1 = plt.gcf()
                fig1.savefig(model.name + "_tr+val_acc_" + filename + ts + '.png')
                plt.figure()
                plt.plot(epochs, loss, 'b', label='Training loss')
                plt.plot(epochs, val_loss, 'r', label='Validation loss')
                plt.title('Training and validation loss')
                plt.legend()

                # plt.figure()
                fig2 = plt.gcf()
                fig2.savefig(model.name + "_tr+val_loss" + filename + ts + '.png')

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

                # EVAL PARAMS FOR CONFUSION MATRIX
                y_test_val = list(evaldf_val.Sentiment)
                predictions = model.predict(x_val, verbose=1, batch_size=BATCH_SIZE)
                y_pred_val = [decode_sentiment(score, include_neutral=False) for score in predictions]

                cnf_matrix = confusion_matrix(y_test_val, y_pred_val, labels=["positive", "negative"])

                # print("printing cnf_matrix...",cnf_matrix)
                plt.figure(figsize=(12, 12))
                plot_confusion_matrix(cnf_matrix, classes=evaldf.Sentiment.unique(), title="Confusion matrix")
                fig3 = plt.gcf()
                plt.show()

                fig3.savefig(model.name + "cm_val_" + filename + ts + '.png')

                print(classification_report(y_test_val, y_pred_val))
                score_acc = accuracy_score(y_test_val, y_pred_val)
                print("printing acc score: ", score_acc)

                report_val = classification_report(y_test_val, y_pred_val, output_dict=True)

                reportData = pd.DataFrame(report_val).transpose()
                reportDataName = model.name + "_classificationReport_val_" + filename + ts + ".csv"
                reportData.to_csv(reportDataName, index=False)

                def testing_metrics(model):
                    y_test = list(evaldf_test.Sentiment)
                    predictions = model.predict(x_test)
                    y_pred = [decode_sentiment(score, include_neutral=False) for score in predictions]

                    cnf_matrix_test = confusion_matrix(y_test, y_pred, labels=['positive', 'negative'])

                    plt.figure(figsize=(12, 12))
                    plot_confusion_matrix(cnf_matrix_test, classes=evaldf.Sentiment.unique(), title="Confusion matrix")
                    fig_cm_test = plt.gcf()
                    plt.show()

                    fig_cm_test.savefig(model.name + "cm_test_" + filename + ts + '.png')

                    report_test = classification_report(y_test, y_pred, output_dict=True)

                    reportData = pd.DataFrame(report_test).transpose()
                    reportDataName = model.name + "_classificationReport_test_" + filename + ts + ".csv"
                    reportData.to_csv(reportDataName, index=False)

                #testing_metrics(model)
            # return model, model_1, model_2, tokenizer
            return model_0, tokenizer  # model_3, model_4,

        amazing2()
