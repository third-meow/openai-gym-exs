
import pickle
import numpy as np
from train_data_setup import create_train_data
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Activation

mdl = keras.models.Sequential()
mdl.add(Dense(128, input_dim=4, activation=tf.nn.relu))
mdl.add(Dense(128, activation=tf.nn.relu))
mdl.add(Dense(128, activation=tf.nn.relu))
mdl.add(Dense(1, activation=tf.nn.softmax))

mdl.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
           )

xtrain, ytrain, xtest, ytest = create_train_data() 

xtrain = keras.utils.normalize(xtrain, axis=1)
xtest = keras.utils.normalize(xtest, axis=1)


mdl.fit(xtrain, ytrain, epochs=8)
print(mdl.evaluate(xtest, ytest, verbose=1))

mdl.save('latest_model.h5')
