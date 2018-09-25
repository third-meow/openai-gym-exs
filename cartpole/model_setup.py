
import pickle
import numpy as np
from train_data_setup import create_train_data
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Activation

mdl = keras.models.Sequential()
mdl.add(Dense(16, input_dim=2, activation='relu'))
mdl.add(Dense(16, activation='sigmoid'))
mdl.add(Dense(16, activation='sigmoid'))
mdl.add(Dense(2, activation='softmax'))

mdl.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
           )

xtrain, ytrain, xtest, ytest = create_train_data() 

print(xtrain.shape)
mdl.fit(xtrain, ytrain, epochs=8)
print(mdl.evaluate(xtest, ytest, verbose=1))

mdl.save('latest_model.h5')
