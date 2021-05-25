# DataFrame
import pandas as pd
# Matplotlib plots
import matplotlib.pyplot as plt
# Sci-kit learn metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, auc, precision_recall_curve, roc_curve
# Sci-kit learn Kfold
from sklearn.model_selection import KFold
# Pre-processing
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Keras Models
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.losses import binary_crossentropy
# Callbacks
from keras.callbacks import EarlyStopping
# Natural Language Toolkit
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
# Word2vec
import gensim
# Utility
import re
import numpy as np
import os
import pickle
import itertools
# Tensorflow
import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disable CUDA for testing, comment out to enable
# debugging GPU:
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
# PARAMS
EPOCHS = 35
DATASIZE = 500000
# Directory
# dirPath = "C:\\Users\\mail\\PycharmProjects\\\MLNLP\\Main\\Data\\Labelled"
# savepathMetrics = "C:\\Users\\mail\\PycharmProjects\\\MLNLP\\Main\\Code\\save\\FinalTrainingMetrics"
# savepathModels = "C:\\Users\\mail\\PycharmProjects\\\MLNLP\\Main\\Code\\save\\FinalTrainingModels"
filename = "Combined_data.csv"
# dirPath_our_labels = "C:\\Users\\mail\\PycharmProjects\\\MLNLP\\Main\\Data\\manually_labelled"
# filename_our_labels = "../../../../Documents/Scraped_merged_-_Updated_.csv"
# Datasets
# file = os.path.join(dirPath, filename)
df = pd.read_csv("../input/data-bp/Combined_data.csv", encoding="ISO-8859-1", names=['text', 'target'], skiprows=1)
# shuffle data
df = df.sample(frac=1)
# only read 1/10th of data
print("df size:", df.index.size)
df = df.drop(df.index[DATASIZE:df.index.size])
print("df reduced size:", df.index.size)

decode_map = {"negative": 0, "positive": 1}

# our_labels = os.path.join(dirPath_our_labels, filename_our_labels)
df_our_labels = pd.read_csv("../input/data-bp/Scraped_merged_-_Updated_.csv", encoding="ISO-8859-1",
                            names=['text', 'ticks', 'target', 'score', 'date', 'URL'], skiprows=1)


# helpful functions
def listToText(l, name):
    with open((name + ".txt"), "w") as filehandle:
        for listitem in str(l):
            filehandle.write(listitem)


def diff(positive, negative):
    # compares size of negative/positive classes and returns even sizes
    if len(positive) > len(negative):
        difference = len(positive) - len(negative)
        print('pos>neg, evening')
        return len(positive) - difference

    elif len(negative) > len(positive):
        difference = len(negative) - len(positive)
        print('neg>pos, evening')
        return len(negative) - difference
    else:
        print('classes are even: positive count = ', len(positive), ', negative count = ', len(negative))


def decode_sentiment(label):
    return decode_map[label]


def preprocess(text, stem=False):  # false or true determines whether the method uses stemmer for tokens.
    # Remove link,user and special characters
    text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)  # String with all tokens seperated with ' '


def decode_prediction(score, include_neutral=True):
    if include_neutral:
        label = 'neutral'
        if score <= 0.4:
            label = 'negative'
        elif score >= 0.7:
            label = 'positive'

        return label
    else:
        return 0 if score < 0.5 else 1


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


# remove neutral
df = df[df.target != "neutral"]

# undersample to even pos/neg ratio
print("Undersampling..")

positive = df[df['target'] == "positive"]
negative = df[df['target'] == "negative"]
if len(positive) != len(negative):
    positive = positive[:diff(positive, negative)]
    negative = negative[:diff(positive, negative)]
    df = pd.concat([positive, negative])
# preprocessing
print("Preprocessing...")
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
df.target = df.target.apply(lambda x: decode_sentiment(x))
df_our_labels.target = df_our_labels.target.apply(lambda x: decode_sentiment(x))
print(df.target)
df.text = df.text.apply(lambda x: preprocess(x))  # text cleaning with Regedit
# split into train, validation and test sets
# dfTrain, dfTest = train_test_split(df, test_size=1 - 0.8, random_state=42)
# print("Test dataset size:", len(dfTest))
# print('Train dataset size: ', len(dfTrain))
dfTrain = df
dfTest = df

# word2vec
docs = []
for t in dfTrain.text:
    docs.append(t.split())
w2v_model = gensim.models.word2vec.Word2Vec(vector_size=300, window=7, min_count=10, workers=8)
w2v_model.build_vocab(docs)  # prepare the model vocabulary
words = w2v_model.wv  # Access the words in vocabulary with .keys()
vocab_size = len(words)
print("Vocab size", vocab_size)  # Print length of vocabulary
w2v_model.train(docs, total_examples=len(docs), epochs=32)
# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dfTrain.text)
vocab_size = len(tokenizer.word_index) + 1
print("Total words", vocab_size)
# tokenize text
x_train = pad_sequences(tokenizer.texts_to_sequences(dfTrain.text), maxlen=300)
x_test = pad_sequences(tokenizer.texts_to_sequences(dfTest.text), maxlen=300)
x_our_labels = pad_sequences(tokenizer.texts_to_sequences(df_our_labels.text), maxlen=300)
# # LabelEncoder
encoder = LabelEncoder()
encoder.fit(dfTrain.target.tolist())
# encode labels to targets
y_train = encoder.transform(dfTrain.target.tolist())
y_test = encoder.transform(dfTest.target.tolist())
y_our_labels = encoder.transform(df_our_labels.target.tolist())
# reshape targets
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_our_labels = y_our_labels.reshape(-1, 1)
# create embedding matrix
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
# embedding layer for neural networks
embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=300, trainable=False)
# callbacks
callbacks = [EarlyStopping(monitor='accuracy', min_delta=0.00001, patience=20, restore_best_weights=True)]
# hyperparameter tuning
batch_sizes = [512, 1024, 8192]
learning_rates = [0.01, 0.001]
layer_sizes = [50, 100]
# K-fold validation
# merge inputs and targets
inputs = np.concatenate((x_train, x_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)
# define K-fold cross validator
kfold = KFold(n_splits=3, shuffle=True)

# ------------------------------------- LSTM ---------------------------------
# scores lists
scoresLSTM = []
scoresLSTM_our_labels = []
acc_per_fold_LSTM = []
acc_LSTM_our_labels = []
loss_per_fold_LSTM = []
loss_LSTM_our_labels = []
# K-fold cross validation model evaluation
fold_no = 1

# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
os.mkdir('saved')
os.chdir('saved')
with tpu_strategy.scope():
    for train, test, in kfold.split(inputs, targets):
        # nested forloops for training models with 16 different hyperparameter setups
        for b in batch_sizes:
            for lr in learning_rates:
                for ls in layer_sizes:
                    # model initialization
                    model_LSTM = Sequential(name='LSTM')
                    model_LSTM.add(embedding_layer)
                    model_LSTM.add(Dropout(0.5))
                    model_LSTM.add(LSTM(ls, dropout=0.2, recurrent_dropout=0.2))
                    model_LSTM.add(Dense(1, activation='sigmoid'))
                    model_LSTM.compile(loss=binary_crossentropy, optimizer=(tf.keras.optimizers.Adam(learning_rate=lr)),
                                       metrics=['accuracy'])
                    hyperparams = "_bs" + str(b) + "_lr" + str(lr) + "ls" + str(ls) + "dl"
                    print("Training for model " + hyperparams)
                    print("Training for fold " + str(fold_no))
                    history = model_LSTM.fit(inputs[train], targets[train], batch_size=b, epochs=EPOCHS, verbose=1,
                                             callbacks=callbacks)
                    # os.chdir(savepathModels)
                    # save model .h5 and .pkl files
                    model_LSTM.save(model_LSTM.name + hyperparams + '.h5')
                    pickle.dump(tokenizer, open('tokenizer' + model_LSTM.name + hyperparams, "wb"), protocol=0)
                    print(hyperparams + " model saved")
                    # Metrics
                    # os.chdir(savepathMetrics)
                    # append eval scores to scores list
                    scoresLSTM.append(model_LSTM.evaluate(inputs[test], targets[test], b))
                    scoresLSTM_our_labels.append(model_LSTM.evaluate(x_our_labels, y_our_labels, b))
                    # create scores for k-fold eval
                    LSTMscores = model_LSTM.evaluate(inputs[test], targets[test], b)
                    LSTM_our_labels_scores = model_LSTM.evaluate(x_our_labels, y_our_labels, b)
                    print(
                        f'Score for fold {fold_no}: {model_LSTM.metrics_names[0]} of {LSTMscores[0]}; {model_LSTM.metrics_names[1]} of {LSTMscores[1] * 100}%')
                    acc_per_fold_LSTM.append(LSTMscores[1] * 100)
                    loss_per_fold_LSTM.append(LSTMscores[0])
                    print(
                        f'Score for our labels: {model_LSTM.metrics_names[0]} of {LSTM_our_labels_scores[0]}; {model_LSTM.metrics_names[1]} of {LSTM_our_labels_scores[1] * 100}%')
                    acc_LSTM_our_labels.append(LSTM_our_labels_scores[1] * 100)
                    loss_LSTM_our_labels.append(LSTM_our_labels_scores[0])
                    # create predictions
                    print("Making Predictions")
                    pred = model_LSTM.predict(inputs[test], verbose=1, batch_size=b)
                    pred = [decode_prediction(score, include_neutral=False) for score in pred]
                    pred_our_labels = model_LSTM.predict(x_our_labels, verbose=1, batch_size=b)
                    pred_our_labels = [decode_prediction(score, include_neutral=False) for score in pred_our_labels]
                    # training accuracy/loss graphs
                    acc = history.history['accuracy']
                    loss = history.history['loss']
                    epochs = range(len(acc))
                    plt.plot(epochs, acc, 'b', label='Training acc')
                    plt.title('Training accuracy')
                    plt.legend()
                    figLSTM_acc = plt.gcf()
                    figLSTM_acc.savefig(model_LSTM.name + "_tr_acc_" + filename + '.png')
                    plt.figure()
                    plt.plot(epochs, loss, 'b', label='Training loss')
                    plt.title('Training and validation loss')
                    plt.legend()
                    figLSTM_loss = plt.gcf()
                    figLSTM_loss.savefig(model_LSTM.name + "_tr_loss" + filename + '.png')
                    plt.show()
                    # Classification Reports
                    cr = classification_report(targets[test], pred, output_dict=True)
                    cr_ours = classification_report(list(y_our_labels), pred_our_labels, output_dict=True)
                    print("Classification Report from test of original dataset:")
                    print(cr)
                    print("Classification Report from test of our labelled data:")
                    print(cr_ours)
                    reportData = pd.DataFrame(cr).transpose()
                    reportData_our_labels = pd.DataFrame(cr_ours).transpose()
                    reportDataName = model_LSTM.name + "_CR_" + hyperparams + filename + ".csv"
                    reportDataName_our_labels = model_LSTM.name + "_CR_our_labels_" + hyperparams + filename + ".csv"
                    reportData.to_csv(reportDataName, index=False)
                    reportData_our_labels.to_csv(reportDataName_our_labels, index=False)
                    # Confusion Matrices
                    cm = confusion_matrix(list(targets[test]), pred, labels=[1, 0])
                    cm_our_labels = confusion_matrix(list(y_our_labels), pred_our_labels, labels=[1, 0])
                    plt.figure(figsize=(12, 12))
                    plot_confusion_matrix(cm, classes=['positive', 'negative'],
                                          title="LSTM Confusion matrix" + "\n" + hyperparams)
                    fig1 = plt.gcf()
                    fig1.savefig(model_LSTM.name + 'CM' + hyperparams + filename + '.png')
                    plt.show()
                    plt.figure(figsize=(12, 12))
                    plot_confusion_matrix(cm_our_labels, classes=['positive', 'negative'],
                                          title="LSTM Confusion Matrix (own labels)" + "\n" + hyperparams)
                    fig2 = plt.gcf()
                    fig2.savefig(model_LSTM.name + 'CM_our_labels_' + hyperparams + filename + '.png')
                    plt.show()
                    # AUC and ROC/AUC Curves
                    precision, recall, thresholds = precision_recall_curve(list(targets[test]), pred)
                    auc_lstm = auc(recall, precision)
                    print('LSTM' + hyperparams + "auc: %.2f" % (auc_lstm))
                    false_pos_rate, true_pos_rate, thresholds = roc_curve(list(targets[test]), pred)
                    fig_auc_lstm = plt.gcf()
                    plt.plot(false_pos_rate, true_pos_rate, marker='.', label='LSTM')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.legend()
                    plt.show()
                    fig_auc_lstm.savefig(model_LSTM.name + 'AUC_ROC_' + hyperparams + filename + '.png')
                    precision, recall, thresholds = precision_recall_curve(list(y_our_labels), pred_our_labels)
                    auc_lstm_ours = auc(recall, precision)
                    print('LSTM_our_labels' + hyperparams + "auc: %.2f" % (auc_lstm_ours))
                    false_pos_rate_ours, true_pos_rate_ours, thresholds_ours = roc_curve(list(y_our_labels),
                                                                                         pred_our_labels)
                    fig_auc_lstm_ours = plt.gcf()
                    plt.plot(false_pos_rate, true_pos_rate, marker='.', label='LSTM')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.legend()
                    plt.show()
                    fig_auc_lstm_ours.savefig(model_LSTM.name + 'AUC_ROC_our_labels_' + hyperparams + filename + '.png')
        # Increase fold number
        fold_no += 1
    # os.chdir(savepathMetrics)
    listToText(scoresLSTM, "scoresLSTM")
    listToText(scoresLSTM_our_labels, "scoresLSTM_our_labels")
    listToText(acc_per_fold_LSTM, "acc_per_fold_LSTM")
    listToText(acc_LSTM_our_labels, "acc_per_fold_LSTM_our_labels")
    listToText(loss_per_fold_LSTM, "loss_per_fold_LSTM")
    listToText(loss_per_fold_LSTM, "loss_per_fold_LSTM")
    listToText(loss_LSTM_our_labels, "loss_per_fold_LSTM_our_labels")
    # ------------------------------------- GRU ---------------------------------
    # scores lists
    scoresGRU = []
    scoresGRU_our_labels = []
    acc_per_fold_GRU = []
    acc_per_fold_GRU_our_labels = []
    loss_per_fold_GRU = []
    loss_per_fold_GRU_our_labels = []
    # K-fold cross validation model evaluation
    fold_no_gru = 1
    for train, test in kfold.split(inputs, targets):
        # nested forloops for training models with 16 different hyperparameter setups
        for b in batch_sizes:
            for lr in learning_rates:
                for ls in layer_sizes:
                    # model initialization
                    model_GRU = Sequential(name='GRU')
                    model_GRU.add(embedding_layer)
                    model_GRU.add(Dropout(0.5))
                    model_GRU.add(GRU(ls))
                    model_GRU.add(Dense(1, activation='sigmoid'))
                    model_GRU.compile(loss=binary_crossentropy, optimizer=(tf.keras.optimizers.Adam(learning_rate=lr)),
                                      metrics=['accuracy'])
                    hyperparams = "_bs" + str(b) + "_lr" + str(lr) + "ls" + str(ls) + "dl"
                    print("Training for model " + hyperparams)
                    print("Training for fold " + str(fold_no_gru))
                    history = model_GRU.fit(inputs[train], targets[train], batch_size=b, epochs=EPOCHS, verbose=1,
                                            callbacks=callbacks)
                    # os.chdir(savepathModels)
                    model_GRU.save(model_GRU.name + hyperparams + '.h5')
                    pickle.dump(tokenizer, open('tokenizer' + model_GRU.name + hyperparams, "wb"), protocol=0)
                    # Metrics
                    # os.chdir(savepathMetrics)
                    scoresGRU.append(model_GRU.evaluate(inputs[test], targets[test], b))
                    scoresGRU_our_labels.append(model_GRU.evaluate(x_our_labels, y_our_labels, b))
                    # create scores for k-fold eval
                    GRUscores = model_GRU.evaluate(inputs[test], targets[test], b)
                    GRU_our_labels_scores = model_GRU.evaluate(x_our_labels, y_our_labels, b)
                    print(
                        f'Score for fold {fold_no}: {model_GRU.metrics_names[0]} of {GRUscores[0]}; {model_GRU.metrics_names[1]} of {GRUscores[1] * 100}%')
                    acc_per_fold_GRU.append(GRUscores[1] * 100)
                    loss_per_fold_GRU.append(GRUscores[0])
                    print(
                        f'Score for our labels: {model_GRU.metrics_names[0]} of {GRU_our_labels_scores[0]}; {model_GRU.metrics_names[1]} of {GRU_our_labels_scores[1] * 100}%')
                    acc_per_fold_GRU_our_labels.append(GRU_our_labels_scores[1] * 100)
                    loss_per_fold_GRU_our_labels.append(GRU_our_labels_scores[0])
                    # Increase fold number
                    fold_no = fold_no + 1
                    # create predictions
                    pred = model_GRU.predict(inputs[test], verbose=1, batch_size=b)
                    pred = [decode_prediction(score, include_neutral=False) for score in pred]
                    pred_our_labels = model_GRU.predict(x_our_labels, verbose=1, batch_size=b)
                    pred_our_labels = [decode_prediction(score, include_neutral=False) for score in pred_our_labels]
                    # training accuracy/loss graphs
                    acc = history.history['accuracy']
                    loss = history.history['loss']
                    epochs = range(len(acc))
                    plt.plot(epochs, acc, 'b', label='Training acc')
                    plt.title('Training accuracy')
                    plt.legend()
                    figGRU_acc = plt.gcf()
                    figGRU_acc.savefig(model_GRU.name + "_tr_acc_" + filename + '.png')
                    plt.figure()
                    plt.plot(epochs, loss, 'b', label='Training loss')
                    plt.title('Training and validation loss')
                    plt.legend()
                    figGRU_loss = plt.gcf()
                    figGRU_loss.savefig(model_GRU.name + "_tr_loss" + filename + '.png')
                    plt.show()
                    # Classification Reports
                    cr = classification_report(targets[test], pred, output_dict=True)
                    cr_ours = classification_report(y_our_labels, pred_our_labels, output_dict=True)
                    print("Classification Report from test of original dataset:")
                    print(cr)
                    print("Classification Report from test of our labelled data:")
                    print(cr_ours)
                    reportData = pd.DataFrame(cr).transpose()
                    reportData_our_labels = pd.DataFrame(cr_ours).transpose()
                    reportDataName = model_GRU.name + "_classificationReport_" + hyperparams + filename + ".csv"
                    reportDataName_our_labels = model_GRU.name + "_CR_our_labels_" + hyperparams + filename + ".csv"
                    reportData.to_csv(reportDataName, index=False)
                    reportData_our_labels.to_csv(reportDataName_our_labels, index=False)
                    # Confusion Matrices
                    cm = confusion_matrix(list(targets[test]), pred, labels=[1, 0])
                    cm_our_labels = confusion_matrix(y_our_labels, pred_our_labels, labels=[1, 0])
                    plt.figure(figsize=(12, 12))
                    fig3 = plt.gcf()
                    fig3.savefig(model_GRU.name + 'CM' + hyperparams + filename + '.png')
                    plot_confusion_matrix(cm, classes=['positive', 'negative'],
                                          title="GRU Confusion matrix" + "\n" + hyperparams)
                    plt.show()
                    plt.figure(figsize=(12, 12))
                    plot_confusion_matrix(cm_our_labels, classes=['positive', 'negative'],
                                          title="GRU Confusion Matrix (own labels)" + "\n" + hyperparams)
                    fig4 = plt.gcf()
                    fig4.savefig(model_GRU.name + 'CM_our_labels_' + hyperparams + filename + '.png')
                    plt.show()
                    # AUC and AUC/ROC Curves
                    precision, recall, _ = precision_recall_curve(list(targets[test]), pred)
                    auc_GRU = auc(recall, precision)
                    print('GRU' + hyperparams + "auc: %.2f" % (auc_GRU))
                    false_pos_rate, true_pos_rate, thresholds = roc_curve(y_our_labels, pred_our_labels)
                    fig_auc_GRU = plt.gcf()
                    plt.plot(false_pos_rate, true_pos_rate, marker='.', label='GRU')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.legend()
                    plt.show()
                    fig_auc_GRU.savefig(model_GRU.name + 'AUC_ROC_' + hyperparams + filename + '.png')
                    precision, recall, _ = precision_recall_curve(y_our_labels, pred_our_labels)
                    auc_GRU_our_labels = auc(recall, precision)
                    print('GRU_our_labels' + hyperparams + "auc: %.2f" % (auc_GRU_our_labels))
                    false_pos_rate_ours, true_pos_rate_ours, thresholds_ours = roc_curve(y_our_labels, pred_our_labels)
                    fig_auc_GRU_ours = plt.gcf()
                    plt.plot(false_pos_rate, true_pos_rate, marker='.', label='GRU')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.legend()
                    plt.show()
                    fig_auc_GRU_ours.savefig(model_GRU.name + 'AUC_ROC_our_labels_' + hyperparams + filename + '.png')
    # Increase fold number
    fold_no_gru += 1
    # os.chdir(savepathMetrics)
    listToText(scoresGRU, "scoresGRU")
    listToText(scoresGRU_our_labels, "scoresGRU_our_labels")
    listToText(acc_per_fold_GRU, "acc_per_fold_GRU")
    listToText(acc_per_fold_GRU_our_labels, "acc_per_fold_GRU_our_labels")
    listToText(loss_per_fold_GRU, "loss_per_fold_GRU")
    listToText(loss_per_fold_GRU_our_labels, "loss_per_fold_GRU_our_labels")

    # ------------------------------------- Bidirectional LSTM ---------------------------------
    # scores lists
    scoresBi_LSTM = []
    scoresBi_LSTM_our_labels = []
    acc_per_fold_Bi_LSTM = []
    acc_per_fold_Bi_LSTM_our_labels = []
    loss_per_fold_Bi_LSTM = []
    loss_per_fold_Bi_LSTM_our_labels = []
    # K-fold cross validation model evaluation
    fold_no_bi_lstm = 1
    for train, test in kfold.split(inputs, targets):
        # nested forloops for training models with 16 different hyperparameter setups
        for b in batch_sizes:
            for lr in learning_rates:
                for ls in layer_sizes:
                    # model initialization
                    Bi_LSTM = Sequential(name='Bi_LSTM')
                    Bi_LSTM.add(embedding_layer)
                    Bi_LSTM.add(Dropout(0.5))
                    Bi_LSTM.add(Bidirectional(LSTM(ls, dropout=0.2, recurrent_dropout=0.2)))
                    Bi_LSTM.add(Dense(1, activation='sigmoid'))
                    Bi_LSTM.compile(loss=binary_crossentropy, optimizer=(tf.keras.optimizers.Adam(learning_rate=lr)),
                                    metrics=['accuracy'])
                    hyperparams = "_bs" + str(b) + "_lr" + str(lr) + "ls" + str(ls) + "dl"
                    print("Training for model " + hyperparams)
                    print("Training for fold " + str(fold_no_bi_lstm))
                    history = Bi_LSTM.fit(inputs[train], targets[train], batch_size=b, epochs=EPOCHS, verbose=1,
                                          callbacks=callbacks)
                    # os.chdir(savepathModels)
                    Bi_LSTM.save(Bi_LSTM.name + hyperparams + '.h5')
                    pickle.dump(tokenizer, open('tokenizer' + Bi_LSTM.name + hyperparams, "wb"), protocol=0)
                    # Metrics
                    # os.chdir(savepathMetrics)
                    scoresBi_LSTM.append(Bi_LSTM.evaluate(inputs[test], targets[test], b))
                    scoresBi_LSTM_our_labels.append((Bi_LSTM.evaluate(x_our_labels, y_our_labels)))
                    # create scores for k-fold eval
                    Bi_LSTMscores = Bi_LSTM.evaluate(inputs[test], targets[test], b)
                    Bi_LSTM_our_labels_scores = Bi_LSTM.evaluate(x_our_labels, y_our_labels, b)
                    print(
                        f'Score for fold {fold_no_bi_lstm}: {Bi_LSTM.metrics_names[0]} of {Bi_LSTMscores[0]}; {Bi_LSTM.metrics_names[1]} of {Bi_LSTMscores[1] * 100}%')
                    acc_per_fold_LSTM.append(Bi_LSTMscores[1] * 100)
                    loss_per_fold_LSTM.append(Bi_LSTMscores[0])
                    print(
                        f'Score for our labels: {Bi_LSTM.metrics_names[0]} of {Bi_LSTM_our_labels_scores[0]}; {Bi_LSTM.metrics_names[1]} of {Bi_LSTM_our_labels_scores[1] * 100}%')
                    acc_per_fold_Bi_LSTM_our_labels.append(Bi_LSTM_our_labels_scores[1] * 100)
                    loss_per_fold_Bi_LSTM_our_labels.append(Bi_LSTM_our_labels_scores[0])
                    # create predictions
                    pred = Bi_LSTM.predict(inputs[test], verbose=1, batch_size=b)
                    pred = [decode_prediction(score, include_neutral=False) for score in pred]
                    pred_our_labels = Bi_LSTM.predict(x_our_labels, verbose=1, batch_size=b)
                    pred_our_labels = [decode_prediction(score, include_neutral=False) for score in pred_our_labels]
                    # training accuracy/loss graphs
                    acc = history.history['accuracy']
                    loss = history.history['loss']
                    epochs = range(len(acc))
                    plt.plot(epochs, acc, 'b', label='Training acc')
                    plt.title('Training accuracy')
                    plt.legend()
                    figBi_LSTM_acc = plt.gcf()
                    figBi_LSTM_acc.savefig(Bi_LSTM.name + "_tr_acc_" + filename + '.png')
                    plt.figure()
                    plt.plot(epochs, loss, 'b', label='Training loss')
                    plt.title('Training and validation loss')
                    plt.legend()
                    figBi_LSTM_loss = plt.gcf()
                    figBi_LSTM_loss.savefig(Bi_LSTM.name + "_tr_loss" + filename + '.png')
                    plt.show()
                    # Classification Reports
                    cr = classification_report(targets[test], pred, output_dict=True)
                    cr_ours = classification_report(y_our_labels, pred_our_labels, output_dict=True)
                    print("Classification Report from test of original dataset:")
                    print(cr)
                    print("Classification Report from test of our labelled data:")
                    print(cr_ours)
                    reportData = pd.DataFrame(cr).transpose()
                    reportData_our_labels = pd.DataFrame(cr_ours).transpose()
                    reportDataName = Bi_LSTM.name + "_classificationReport_" + hyperparams + filename + ".csv"
                    reportDataName_our_labels = Bi_LSTM.name + "_CR_our_labels_" + hyperparams + filename + ".csv"
                    reportData.to_csv(reportDataName, index=False)
                    reportData_our_labels.to_csv(reportDataName_our_labels, index=False)
                    # Confusion matrices
                    cm = confusion_matrix(list(targets[test]), pred, labels=[1, 0])
                    cm_our_labels = confusion_matrix(y_our_labels, pred_our_labels, labels=[1, 0])
                    plt.figure(figsize=(12, 12))
                    fig5 = plt.gcf()
                    fig5.savefig(Bi_LSTM.name + 'CM' + hyperparams + filename + '.png')
                    plot_confusion_matrix(cm, classes=['positive', 'negative'],
                                          title="Bi-LSTM Confusion matrix" "\n" + hyperparams)
                    plt.show()
                    plt.figure(figsize=(12, 12))
                    plot_confusion_matrix(cm_our_labels, classes=['positive', 'negative'],
                                          title="Bi-LSTM Confusion matrix (own labels)" "\n" + hyperparams)
                    fig6 = plt.gcf()
                    fig6.savefig(Bi_LSTM.name + 'CM_our_labels' + hyperparams + filename + '.png')
                    plt.show()
                    # AUC and AUC/ROC Curves
                    precision, recall, _ = precision_recall_curve(list(targets[test]), pred)
                    auc_Bi_LSTM = auc(recall, precision)
                    print('Bi_LSTM' + hyperparams + "auc: %.2f" % (auc_Bi_LSTM))
                    false_pos_rate, true_pos_rate, thresholds = roc_curve(list(targets[test]), pred)
                    fig_auc_Bi_LSTM = plt.gcf()
                    plt.plot(false_pos_rate, true_pos_rate, marker='.', label='Bi-LSTM')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.legend()
                    plt.show()
                    fig_auc_Bi_LSTM.savefig(Bi_LSTM.name + 'AUC_ROC_' + hyperparams + filename + '.png')
                    precision, recall, thresholds = precision_recall_curve(y_our_labels, pred_our_labels)
                    auc_Bi_LSTM_our_labels = auc(recall, precision)
                    print('Bi_LSTM_our_labels' + hyperparams + "auc: %.2f" % (auc_Bi_LSTM_our_labels))
                    false_pos_rate_ours, true_pos_rate_ours, thresholds_ours = roc_curve(y_our_labels, pred_our_labels)
                    fig_auc_Bi_LSTM_ours = plt.gcf()
                    plt.plot(false_pos_rate, true_pos_rate, marker='.', label='Bi-LSTM')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.legend()
                    plt.show()
                    fig_auc_Bi_LSTM_ours.savefig(Bi_LSTM.name + 'AUC_ROC_our_labels_' + hyperparams + filename + '.png')
    # Increase fold number
    fold_no_bi_lstm += 1
    # os.chdir(savepathMetrics)
    listToText(scoresBi_LSTM, "scoresBi_LSTM")
    listToText(scoresBi_LSTM_our_labels, "scoresBi_LSTM_our_labels")
    listToText(acc_per_fold_Bi_LSTM, "acc_per_fold_Bi_LSTM")
    listToText(acc_per_fold_Bi_LSTM_our_labels, "acc_per_fold_Bi_LSTM_our_labels")
    listToText(loss_per_fold_Bi_LSTM, "loss_per_fold_Bi_LSTM")
    listToText(loss_per_fold_Bi_LSTM_our_labels, "loss_per_fold_Bi_LSTM_our_labels")
