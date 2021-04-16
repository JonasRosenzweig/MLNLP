import MethodHandler

# MethodHandler.loadModel()
#
# model = MethodHandler.loadModel.model
# tokenizer = MethodHandler.loadModel.tokenizer
# MethodHandler.predictSentiment("Titanic is sinking", model=model, tokenizer=tokenizer)
#
# MethodHandler.csvPredicter(CSVPATH='C:\\Users\\HE400\\PycharmProjects\\MLNLP2\\Tutorials_and_References\\notebookGold\\MLNLP\\input\\filename.csv', model=model, tokenizer=tokenizer)

# MODEL TRAINED WITHOUT stems


#train model:
trainObj = MethodHandler.trainModel()

m = trainObj[0]
m_1 = trainObj[1]
t = trainObj[2]


