
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import TensorBoard

def build_model():
    mdl = keras.models.Sequential()
    mdl.add(Dense(128, input_dim=4, activation=tf.nn.tanh))
    mdl.add(Dense(128, activation=tf.nn.tanh))
    mdl.add(Dense(128, activation=tf.nn.tanh))
    mdl.add(Dense(1, activation=tf.nn.tanh))

    mdl.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return mdl

def train_model(mdl):
    xtrain = pickle.load(open('saved/xtrain.pickle', 'rb'))
    ytrain = pickle.load(open('saved/ytrain.pickle', 'rb'))
    xtest = pickle.load(open('saved/xtest.pickle', 'rb'))
    ytest = pickle.load(open('saved/ytest.pickle', 'rb'))


    name = 'cartpole-{}'.format(int(time.time()))
    tensorboard = TensorBoard(log_dir='saved/logs/{}'.format(name))

    mdl.fit(xtrain, ytrain, batch_size=16, epochs=4, callbacks=[tensorboard])
    print(mdl.evaluate(xtest, ytest, verbose=1))
    
    return mdl

if __name__ == '__main__':
    mdl = build_model()
    mdl = train_model(mdl)
    mdl.save('saved/latest_model.h5')

