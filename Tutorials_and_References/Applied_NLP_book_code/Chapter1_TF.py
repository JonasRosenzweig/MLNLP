# Applied Natural Language Processing with Python: Implementing Machine Learning
# and Deep Learning Algorithms for Natural Language Processing, by Taweh Beysolow II
# Chapter 1 - TF code example

import tensorflow as tf

# random variables
state_size = 2
n_classes = 2
batch_size = 32
bprop_len = 16

# Creating weights and biases dictionaries
weights = {'input': tf.Variable(tf.random_normal([state_size+1, state_size])),
           'output': tf.Variable(tf.random_normal([state_size,n_classes]))}
biases = {'input': tf.Variable(tf.random_normal([1, state_size])),
          'output': tf.Variable(tf.random_normal([1, n_classes]))}

# Defining placeholders and variables
X = tf.placeholder(tf.float32, [batch_size, bprop_len])
Y = tf.placeholder(tf.int32, [batch_size, bprop_len])
init_state = tf.placeholder(tf.float32, [batch_size, state_size])
input_series = tf.unstack(X, axis=1)
labels = tf.unstack(Y, axis=1)
current_state = init_state
hidden_states = []

# Passing values from one hidden state to the next
for input in input_series:  # Evaluating each input within the series of inputs
    input = tf.reshape(input, [batch_size, 1])  # Reshaping into MxN tensor
    # Concatenating input and current state tensors
    input_state = tf.concat([input, current_state], axis=1)
    # Tanh transformation
    _hidden_state = tf.tanh(tf.add(tf.matmult(input_state, weights['input']),
                                   biases['input']))
    current_state = _hidden_state  # Updating current state
