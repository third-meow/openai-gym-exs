
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard

def build_model():
    mdl = keras.models.Sequential()
    mdl.add(Dense(64, input_dim=12, activation=tf.nn.softplus))
    mdl.add(Dense(64, activation=tf.nn.softplus))
    mdl.add(Dense(32, activation=tf.nn.softplus))
    mdl.add(Dense(32, activation=tf.nn.softplus))
    mdl.add(Dense(1, activation=tf.nn.relu))
    mdl.add(Dense(1, activation=tf.nn.softplus))

    mdl.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return mdl

def train_model(mdl):
    xtrain = pickle.load(open('saved/normal/xtrain.pickle', 'rb'))
    ytrain = pickle.load(open('saved/normal/ytrain.pickle', 'rb'))
    xtest = pickle.load(open('saved/normal/xtest.pickle', 'rb'))
    ytest = pickle.load(open('saved/normal/ytest.pickle', 'rb'))


    name = 'cartpole-{}'.format(int(time.time()))
    tensorboard = TensorBoard(log_dir='saved/logs/{}'.format(name))

    mdl.fit(xtrain, ytrain, batch_size=32, epochs=3, callbacks=[tensorboard])
    print(mdl.evaluate(xtest, ytest, verbose=1))
    count = 0
    for i in range(len(xtest)):
        if int(mdl.predict(np.array([xtest[i]]))) == ytest[i]:
            print(xtest[i])
            count += 1
            if count > 10:
                quit()
    
    return mdl

if __name__ == '__main__':
    mdl = build_model()
    mdl = train_model(mdl)
    mdl.save('saved/latest_model.h5')

