from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.model_selection import train_test_split
import itertools
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

dirPath = "C:\\Users\\mail\\PycharmProjects\\\MLNLP\\Main\\Data\\Labelled"
savepath = "C:\\Users\\mail\\PycharmProjects\\\MLNLP\\Main\\Code\\save\\test"


# METHODS

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

def decodeClass(score):
    if score == 0:
        label = 'positive'
    if score == 1:
        label = 'negative'
    return label


def decode_sentiment(label):
    return decode_map[int(label)]


def diff(positive, negative):
    if len(positive) > len(negative):
        difference = len(positive) - len(negative)
        return len(positive) - difference

    elif len(negative) > len(positive):
        difference = len(negative) - len(positive)
        return len(negative) - difference

    else:
        print('something went wrong..', len(positive), len(negative))


filenames = ["Airline_Tweets.csv", "Financial_news_all-data.csv", "IMDB_Dataset.csv", "Reddit_Data.csv",
             "Scraped_merged_-_Updated_.csv", "Scraped_merged_manually_labelled.csv", "sentiment140.csv",
             "Steam_train.csv", "Twitter_Data.csv"]

for filename in filenames:
    to_save_path = savepath + '\\' + filename.split('.csv')[0]
    os.mkdir(to_save_path)
    os.chdir(to_save_path)

    file = os.path.join(dirPath, filename)
    print(filename)
    # Airline_Tweets.csv
    if filename == "Airline_Tweets.csv":
        # decode_map = {"negative": "NEGATIVE", "neutral": "NEUTRAL", "positive": "POSITIVE"}
        DATASET_COLUMNS = ["tweet_id", "target", "airline_sentiment_confidence", "negativereason",
                           "negativereason_confidence", "airline", "airline_sentiment_gold", "name",
                           "negativereason_gold", "retweet_count",
                           "text", "tweet_coord", "tweet_created", "tweet_location", "user_timezone"]
        df = pd.read_csv(file, encoding='ISO-8859-1', names=DATASET_COLUMNS, skiprows=1)
        df = df[df.target != "neutral"]  # removes neutral

        positive = df[df['target'] == "positive"]
        negative = df[df['target'] == "negative"]

    if len(positive) != len(negative):
        positive = positive[:diff(positive, negative)]
        negative = negative[:diff(positive, negative)]
        df = pd.concat([positive, negative])

    elif filename == "sentiment140.csv":
        DATASET_COLUMNS = ["target", "id", "date", "flag", "user", "text"]
        decode_map = {0: "negative", 4: "positive"}
        df = pd.read_csv(file, encoding='ISO-8859-1', names=DATASET_COLUMNS, skiprows=1)
        df.target = df.target.apply(lambda x: decode_sentiment(x))
        df = df[df.target != "neutral"]

        positive = df[df['target'] == "positive"]
        negative = df[df['target'] == "negative"]
        if len(positive) != len(negative):
            positive = positive[:diff(positive, negative)]
            negative = negative[:diff(positive, negative)]
            df = pd.concat([positive, negative])

    # Scraped_merged_ - _Updated_.csv
    elif filename == "Scraped_merged_-_Updated_.csv":
        DATASET_COLUMNS = ["text", "Ticks", "target", "Score", "Date", "URL"]
        df = pd.read_csv(file, encoding='ISO-8859-1', names=DATASET_COLUMNS, skiprows=1)
        positive = df[df['target'] == "positive"]
        negative = df[df['target'] == "negative"]
        if len(positive) != len(negative):
            positive = positive[:diff(positive, negative)]
            negative = negative[:diff(positive, negative)]
            df = pd.concat([positive, negative])

    # Scraped_merged_manually_labelled.csv
    elif filename == "Scraped_merged_manually_labelled.csv":
        DATASET_COLUMNS = ["text", "Ticks", "target", "Score", "Date", "URL"]
        df = pd.read_csv(file, encoding='ISO-8859-1', names=DATASET_COLUMNS, skiprows=1)
        positive = df[df['target'] == "positive"]
        negative = df[df['target'] == "negative"]
        if len(positive) != len(negative):
            positive = positive[:diff(positive, negative)]
            negative = negative[:diff(positive, negative)]
            df = pd.concat([positive, negative])
    # Financial_news_all
    elif filename == "Financial_news_all-data.csv":
        DATASET_COLUMNS = ["target", "text"]
        df = pd.read_csv(file, encoding='ISO-8859-1', names=DATASET_COLUMNS)
        df = df[df.target != "neutral"]  # removes neutral

        positive = df[df['target'] == "positive"]
        negative = df[df['target'] == "negative"]

        if len(positive) != len(negative):
            positive = positive[:diff(positive, negative)]
            negative = negative[:diff(positive, negative)]
            df = pd.concat([positive, negative])
    # IMDB_Dataset.csv
    elif filename == "IMDB_Dataset.csv":
        DATASET_COLUMNS = ["text", "target"]
        df = pd.read_csv(file, encoding='ISO-8859-1', names=DATASET_COLUMNS, skiprows=1)
        df = df[df.target != "neutral"]  # removes neutral
        decode_map = {1: 'positive', 0: 'negative'}

        positive = df[df['target'] == "positive"]
        negative = df[df['target'] == "negative"]

        if len(positive) != len(negative):
            positive = positive[:diff(positive, negative)]
            negative = negative[:diff(positive, negative)]
            df = pd.concat([positive, negative])

    # Reddit_Data.csv
    elif filename == "Reddit_data.csv":
        decode_map = {-1: "negative", 1: "positive"}
        DATASET_COLUMNS = ["text", "target"]
        df = pd.read_csv(file, encoding='ISO-8859-1', names=DATASET_COLUMNS, skiprows=1)
        df.target = df.target.apply(lambda x: decode_sentiment(x))
        df = df[df.target != "neutral"]  # removes neutral

        positive = df[df['target'] == "positive"]
        negative = df[df['target'] == "negative"]

        if len(positive) != len(negative):
            positive = positive[:diff(positive, negative)]
            negative = negative[:diff(positive, negative)]
            df = pd.concat([positive, negative])

    # Steam_train.csv
    elif filename == "Steam_train.csv":
        DATASET_COLUMNS = ["review_id", "title", "year", "text", "target"]
        decode_map = {1: "negative", 0: "positive"}
        df = pd.read_csv(file, encoding='ISO-8859-1', names=DATASET_COLUMNS, skiprows=1)
        df.target = df.target.apply(lambda x: decode_sentiment(x))
        df = df[df.target != "neutral"]  # removes neutral

        positive = df[df['target'] == "positive"]
        negative = df[df['target'] == "negative"]

        if len(positive) != len(negative):
            positive = positive[:diff(positive, negative)]
            negative = negative[:diff(positive, negative)]
            df = pd.concat([positive, negative])

    # Twitter_data.csv
    elif filename == "Twitter_data.csv":
        DATASET_COLUMNS = ["text", "target"]
        decode_map = {-1: "negative", 1: "positive"}
        df = pd.read_csv(file, encoding='ISO-8859-1', names=DATASET_COLUMNS, skiprows=1)
        df.target = df.target.apply(lambda x: decode_sentiment(x))
        df = df[df.target != "neutral"]  # removes neutral

        positive = df[df['target'] == "positive"]
        negative = df[df['target'] == "negative"]

        if len(positive) != len(negative):
            positive = positive[:diff(positive, negative)]
            negative = negative[:diff(positive, negative)]
            df = pd.concat([positive, negative])

    # encode text and target (X,Y) values to feed into classifiers
    le = LabelEncoder()
    #x_text = le.fit_transform(df.text)
    y_target = le.fit_transform(df.target)
    #x_text = x_text.reshape(-1, 1)
    y_target = y_target.reshape(-1, 1)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df.text)
    x_text = pad_sequences(tokenizer.texts_to_sequences(df.text), maxlen=300)
    #print(len(x_text))


    # test train split
    x_train, x_test, y_train, y_test = train_test_split(x_text, y_target, test_size=0.1, random_state=42)

    # targets' classes
    #y_test_1d = list(y_test)
    #y_test_1d_decoded = [decodeClass(score) for score in y_test_1d]

    # Classification ---

    # Random Forest ----------------------
    Rf = RandomForestClassifier(max_depth=4, random_state=0)
    # "train" clf
    Rf.fit(x_train, y_train)
    # predictions
    Rf_predictions = Rf.predict(x_test)
    # predictions classes
    #y_Rf_pred_test = [decodeClass(score) for score in Rf_predictions]
    # classification report for random forest
    cr = classification_report(y_test, Rf_predictions, output_dict=True)
    print(cr)
    reportData = pd.DataFrame(cr).transpose()
    reportDataName = 'Random_Forest' + "_classificationReport_" + filename + ".csv"
    reportData.to_csv(reportDataName, index=False)

    # confusion matrix for random forest
    plt.figure(figsize=(12, 12))
    Rf_cnf_matrix = confusion_matrix(y_test, Rf_predictions, labels=["positive", "negative"])
    plot_confusion_matrix(Rf_cnf_matrix, classes=['positive', 'negative'], title="Random Forest Confusion Matrix")
    fig1 = plt.gcf()
    fig1.savefig("Random_Forest" + filename + '.png')
    plt.show()

    # Logistic Regression ----------------------
    logreg_model = LogisticRegression(solver='lbfgs')
    # "train" logreg
    logreg_model.fit(x_train, y_train)
    # predictions class probabilities
    logreg_predictions = logreg_model.predict(x_test)
    # targets' classes
    #y_Lg_pred_test = [decodeClass(score) for score in logreg_predictions]
    # classification report logistic regression
    cr = classification_report(y_test, logreg_predictions, output_dict=True)
    print(cr)
    reportData = pd.DataFrame(cr).transpose()
    reportDataName = 'Logistic_regression' + "_classificationReport_" + filename + ".csv"
    reportData.to_csv(reportDataName, index=False)

    # confusion matrix for logistic regression
    logreg_cnf_matrix = confusion_matrix(y_test, logreg_predictions)
    plt.figure(figsize=(12, 12))
    fig2 = plt.gcf()
    fig2.savefig("Logistic_Regression" + filename + '.png')
    plot_confusion_matrix(logreg_cnf_matrix, classes=['positive', 'negative'], title="LogReg Confusion matrix")
    plt.show()

    # AdaBoost ----------------------
    Ab = AdaBoostClassifier(n_estimators=100, random_state=0)
    # "train" adaboost
    Ab.fit(x_train, y_train)
    # predictions classes
    Ab_predictions = Ab.predict(x_test)
    # targets' classes
    #y_Ab_pred_test = [decodeClass(score) for score in Ab_predictions]
    # classification report AdaBoost
    cr = classification_report(y_test, Ab_predictions, output_dict=True)
    print(cr)
    reportData = pd.DataFrame(cr).transpose()
    reportDataName = 'AdaBoost' + "_classificationReport_" + filename + ".csv"
    reportData.to_csv(reportDataName, index=False)
    # confusion matrix for logistic regression
    Ab_cnf_matrix = confusion_matrix(y_test, Ab_predictions)
    plt.figure(figsize=(12, 12))
    fig3 = plt.gcf()
    fig3.savefig('Ada_boost' + filename + '.png')
    plot_confusion_matrix(Ab_cnf_matrix, classes=['positive', 'negative'], title="AdaBoost Confusion matrix")
    plt.show()

    # Naive Bayes ----------------------
    Nb = GaussianNB()
    # "train" Naivebayes
    Nb.fit(x_train, y_train)
    # predictions classes
    Nb_predictions = Nb.predict(x_test)
    # targets' classes
    #y_Nb_pred_test = [decodeClass(score) for score in Nb_predictions]
    # classification report Naivebayes
    cr = classification_report(y_test, Nb_predictions, output_dict=True)
    print(cr)
    reportData = pd.DataFrame(cr).transpose()
    reportDataName = 'Naive_Bayes' + "_classificationReport_" + filename + ".csv"
    reportData.to_csv(reportDataName, index=False)
    # confusion matrix for Naive Bayes
    Nb_cnf_matrix = confusion_matrix(y_test, Nb_predictions)
    plt.figure(figsize=(12, 12))
    fig4 = plt.gcf()
    fig4.savefig('Naive_Bayes' + filename + '.png')
    plot_confusion_matrix(Nb_cnf_matrix, classes=['positive', 'negative'], title="NaiveBayes Confusion matrix")
    plt.show()