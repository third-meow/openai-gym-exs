
import pickle
import numpy as np
from train_data_setup import create_train_data
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Activation

mdl = keras.models.Sequential()
mdl.add(Dense(16, input_dim=4, activation='relu'))
mdl.add(Dense(16, activation='relu'))
mdl.add(Dense(16, activation='relu'))
mdl.add(Dense(1, activation='softmax'))

mdl.compile(
            optimizer='adam',
            loss='binary_crossentropy'
           )

xtrain, ytrain, xtest, ytest = create_train_data 

mdl.fit(xtrain, ytrain, epochs=3, verbose=0)
#print(mdl.evaluate(

mdl.save('latest_model.h5')
