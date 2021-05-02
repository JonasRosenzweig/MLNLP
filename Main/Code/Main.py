import MethodHandler
import tensorflow as tf
# MethodHandler.loadModel()
#
# model = MethodHandler.loadModel.model
# tokenizer = MethodHandler.loadModel.tokenizer
# MethodHandler.predictSentiment("Titanic is sinking", model=model, tokenizer=tokenizer)
#
# MethodHandler.csvPredicter(CSVPATH='C:\\Users\\HE400\\PycharmProjects\\MLNLP2\\Tutorials_and_References\\notebookGold\\MLNLP\\input\\filename.csv', model=model, tokenizer=tokenizer)

# MODEL TRAINED WITHOUT stems

# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
#
#
# with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
#
# #train model:
#     trainObj = MethodHandler.trainModel()
#
#     m = trainObj[0]
#     m_1 = trainObj[1]
#     t = trainObj[2]
print('training')
trainObj = MethodHandler.trainModel()
print('done!')
m = trainObj[0]
m_1 = trainObj[1]
t = trainObj[2]