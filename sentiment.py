
##########################
# Code for Ex. #2 in IDL #
##########################



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import models, layers, optimizers
import tensorflow as tf
import bz2
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import tensorflow.keras.backend as K

import sys
sys.setrecursionlimit(2500)

import os

import loader as ld

train_texts, train_labels, test_texts, test_labels, test_ascii, embedding_matrix, MAX_LENGTH, MAX_FEATURES = ld.get_dataset()

#####################
# Execusion options #
#####################

TRAIN = True

RECR = False # recurrent netowrk (RNN/GRU) or a non-recurrent network

ATTN = True # use attention layer in global sum pooling or not
LSTM = False # use LSTM or otherwise RNN

WEIGHTED = False


# Getting activations from model

def get_act(net, input, name):
  sub_score = [layer for layer in model.layers if name in layer.name][0].output
  # functor = K.function([test_texts]+ [K.learning_phase()], sub_score)

  OutFunc = K.function([net.input], [sub_score])
  return OutFunc([test_texts])[0]


# RNN Cell Code

def RNN(dim ,x):

  # Learnable weights in the cell
  Uh = layers.Dense(dim, use_bias=False, activation='relu')
  Wh = layers.Dense(dim, activation='relu')

  # unstacking the time axis
  x = tf.unstack(x, axis=1)

  H = []

  h = tf.zeros_like(Wh(x[0]))

  for i in range(len(x)):
    # Apply the basic step in each time step

    h = Wh(x[i]) + Uh(h)

    H.append(h)

  H = tf.stack(H, axis=1)

  return h, H

# GRU Cell Code

def GRU(dim ,x):
    # Learnable weights in the cell
    Wzx = layers.Dense(dim, activation='sigmoid')
    Wzh = layers.Dense(dim, use_bias=False, activation='sigmoid')

    Wrx = layers.Dense(dim, activation='sigmoid')
    Wrh = layers.Dense(dim, use_bias=False, activation='sigmoid')

    Wx = layers.Dense(dim, activation='tanh')
    Wh = layers.Dense(dim, use_bias=False, activation='tanh')

    # unstacking the time axis
    x = tf.unstack(x, axis=1)

    H = []

    h = tf.zeros_like(Wx(x[0]))

    for i in range(len(x)):
        z = Wzx(x[i]) + Wzh(h)
        r = Wrx(x[i]) + Wrh(h)
        ht = Wx(x[i]) + Wh(r * h)
        h = (1 - z) * h + z * ht

        H.append(h)

    H = tf.stack(H, axis=1)

    return h, H

# (Spatially-)Restricted Attention Layer
# k - specifies the -k,+k neighbouring words

def restricted_attention(x ,k):
  dim = x.shape[2]

  Wq = layers.Dense(dim)
  Wk = layers.Dense(dim)

  wk = Wk(x)

  paddings = tf.constant([[0, 0 ,], [k, k], [0 ,0]])
  pk = tf.pad(wk, paddings)
  pv = tf.pad(x, paddings)

  keys = []
  vals = []
  for i in range(-k , k +1):
    keys.append(tf.roll(pk ,i ,1))
    vals.append(tf.roll(pv ,i ,1))

  keys = tf.stack(keys ,2)
  keys = keys[: ,k:-k ,: ,:]
  vals = tf.stack(vals ,2)
  vals = vals[: ,k:-k ,: ,:]

  # -- missing code --

  query = Wq(x)
  query = tf.reshape(query, [-1, query.shape[1], query.shape[1], 1])

  dot_product = tf.matmul(keys, query) / np.sqrt(dim)

  atten_weights = tf.nn.softmax(dot_product, name="atten_weights")

  val_out = tf.matmul(atten_weights, vals, True)
  val_out = tf.reshape(val_out, [-1, val_out.shape[1], val_out.shape[3]])

  return x + val_out


# Building Entire Model
def build_model():
    sequences = layers.Input(shape=(MAX_LENGTH,))
    embedding_layer = layers.Embedding(MAX_FEATURES, 100, weights=[embedding_matrix], input_length=MAX_LENGTH,
                                       trainable=False)

    # embedding the words into 100 dim vectors

    x = embedding_layer(sequences)

    if not RECR:

        # non recurrent networks

        if ATTN:
            # attention layer
            x = restricted_attention(x, k=5)

        # word-wise FC layers -- MAKE SURE you have ,name= "sub_score" in the sub_scores step
        # E.g., sub_score = layers.Dense(2,name="sub_score")(x)

        unstacked_x = tf.unstack(x, axis=1)

        layer1 = layers.Dense(20, activation="relu")
        layer2 = layers.Dense(1)
        if WEIGHTED:
            weights_layer = layers.Dense(1)
        sub_score = []
        weights = []

        for i in range(len(unstacked_x)):
            output1 = layer1(unstacked_x[i])
            output2 = layer2(output1)
            if WEIGHTED:
                weights_output = weights_layer(output1)
                weights.append(weights_output)

            sub_score.append(output2)

        # sum / weighted sum
        stacked = tf.stack(sub_score, axis=1, name="sub_score")

        if WEIGHTED:
            weights_vector = tf.nn.softmax(tf.stack(weights, axis=1))
        else:
            weights_vector = tf.ones_like(stacked)

        sum_score = tf.matmul(stacked, weights_vector, transpose_a=True)
        sum_score = tf.reshape(sum_score, (-1, 1))

        # -- missing code --

        # final prediction

        final_prediction = tf.sigmoid(sum_score)

        predictions = final_prediction

    else:
        # recurrent networks
        if LSTM:
            x, _ = GRU(64, x)
        else:
            x, _ = RNN(64, x)

        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        predictions = x

    model = models.Model(inputs=sequences, outputs=predictions)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model


model = build_model()

checkpoint_path = "model_save/cp.ckpt"

if TRAIN:
    print("Training")

    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True)
    print(model.summary())

    model.fit(
        train_texts,
        train_labels,
        batch_size=128,
        epochs=1,
        validation_data=(test_texts, test_labels), callbacks=[cp_callback])
else:
    model.load_weights(checkpoint_path)

#############
# test code #
#############

print("Example Predictions:")
preds = model.predict(test_texts)

if not RECR:
    sub_score = get_act(model, test_texts, "sub_score")

for i in range(4):

    print("-" * 20)

    if not RECR:
        # print words along with their sub_score

        num = min((len(test_ascii[i]), 100))
        for k in range(num):
            print(test_ascii[i][k], sub_score[i][k])

        print("\n")
    else:
        print(test_ascii[i])
        print(preds[i])

    if preds[i] > 0.5:
        print("Positive")
    else:
        print("Negative")
    print("-" * 20)

print('Accuracy score: {:0.4}'.format(accuracy_score(test_labels, 1 * (preds > 0.5))))
print('F1 score: {:0.4}'.format(f1_score(test_labels, 1 * (preds > 0.5))))
print('ROC AUC score: {:0.4}'.format(roc_auc_score(test_labels, preds)))