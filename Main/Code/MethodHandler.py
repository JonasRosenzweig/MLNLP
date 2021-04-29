import os

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import pickle
import pandas as pd
import modell
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc
import time

#savepath = "C:\\Users\\Jonas\\PycharmProjects\\MLNLP\\Main\\Code\\save" #Jonas path home desktop
savepath = "C:\\Users\\mail\\PycharmProjects\\\MLNLP\\Main\\Code\\save" #Jonas path work
#savepath = "C:\\Users\\HE400\\PycharmProjects\\MLNLP_main\\Main\\Code\\save" # Hammi path

def metrics():
    return

def trainModel():
    amazing_obj = modell.amazing()
    m = amazing_obj[0]
    m_1 = amazing_obj[1]
    m_2 = amazing_obj[2]
    m_3 = amazing_obj[3]
    t = amazing_obj[2]
    return m, m_1, m_2, m_3, t             # m represents model and t tokenizer


def saveModel(model, tokenizer, dataset):
    path = savepath
    ts = time.gmtime()
    ts = time.strftime("%Y-%m-%d_%H-%M-%S", ts)
    modelName = model.name+'_'+dataset+'_'+ts+'.h5'
    tokenizerName = model.name+'_'+dataset+'_'+ts+'tokenizer.pkl'

    os.chdir(path)

    model.save(modelName)
    pickle.dump(tokenizer, open(tokenizerName, "wb"), protocol=0)


# def loadModel():
#     model = keras.models.load_model('C:\\Users\\HE400\\PycharmProjects\\MLNLP2\\Tutorials_and_References\\notebookGold\\model.h5')
#     tokenizer = Tokenizer()
#     test_input = pad_sequences(tokenizer.texts_to_sequences(['I love the rain and trump is amazing']), maxlen=300)
#     print(test_input)
#     score = model.predict([test_input])[0]
#     #print(model.summary())
#     model.summary()

def loadModel(path):

    #w2v_model = Word2Vec.load(path + 'model1.w2v')

    model = load_model(path)  # Create function attribute.


    # x_test = pad_sequences(tokenizer.texts_to_sequences(['I love music']), maxlen=300)
    # score = model.predict([x_test])[0]
    # with open(path + 'tokenizer1.pkl', 'rb') as handle: tokenizer = pickle.load(handle)
    # with open(path + 'encoder1.pkl', 'rb') as handle: encoder = pickle.load(handle)
    # print(score)
    return model

def loadTokenizer(path):
    tokenizer = pickle.load(open(path, 'rb'))
    return tokenizer


# loadModel()

def predictSentiment(text, model, tokenizer):
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=300)
    score = model.predict([x_test])[0]
    print(score)
    return score


def decode_sentiment(score, include_neutral=True):
    if include_neutral:
        label = 'NEUTRAL'
        if score <= 0.4:
            label = 'NEGATIVE'
        elif score >= 0.7:
            label = 'POSITIVE'

        return label
    else:
        return 'NEGATIVE' if score < 0.5 else 'POSITIVE'


def csvPredicter(CSVPATH, model, tokenizer):
    print('predictingCSV')
    df = pd.read_csv(CSVPATH, names=['sentence', 'ticks'])
    print(df.sentence)
    print('printing df...', df)

    pred = []
    predScore = []
    for i in df['sentence']:
        score = predictSentiment(i, model=model, tokenizer=tokenizer)
        pred.append(decode_sentiment(score))
        predScore.append(score)

    Table = {'sentence': df['sentence'],
             'pred': pred,
             'score': predScore,
             'ticks': df['ticks']}
    dfPred = pd.DataFrame(Table)
    dfPred.to_csv('daniel.csv')
