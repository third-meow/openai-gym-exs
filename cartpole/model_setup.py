
import pickle
import numpy as np
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


#initial training data
xtrain = np.array([
            [ 0.08961755, 1.17621839,-0.09048343,-1.76697922],
            [ 0.11314191, 1.37223859,-0.12582301,-2.08637252],
            [ 0.14058669, 1.56838708,-0.16755046,-2.41516379],
            [ 0.17195443, 1.76451908,-0.21585374,-2.75427158],
            [-0.04526003,-1.19452086, 0.09759106, 1.81212372],
            [-0.06915045,-1.39058518, 0.13383353, 2.13346707],
            [-0.09696215,-1.58675556, 0.17650287, 2.46432101],
            [-0.12869727,-1.78287383, 0.22578929, 2.80555083]
          ])

ytrain = np.array([0,0,0,0,1,1,1,1])

mdl.fit(xtrain, ytrain, epochs=3, verbose=0)

mdl.save('latest_model.h5')
