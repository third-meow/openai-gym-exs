import pickle
import numpy as np
from keras.utils import normalize

xtrain = pickle.load(open('saved/raw/xtrain.pickle', 'rb'))
ytrain = pickle.load(open('saved/raw/ytrain.pickle', 'rb'))
xtest = pickle.load(open('saved/raw/xtest.pickle', 'rb'))
ytest = pickle.load(open('saved/raw/ytest.pickle', 'rb'))

for train_set in xtrain: 
    train_set = normalize(train_set)
for test_set in xtest:
    test_set = normalize(test_set)

pickle.dump(xtrain, open('saved/normal/xtrain.pickle', 'wb'))
pickle.dump(ytrain, open('saved/normal/ytrain.pickle', 'wb'))
pickle.dump(xtest, open('saved/normal/xtest.pickle', 'wb'))
pickle.dump(ytest, open('saved/normal/ytest.pickle', 'wb'))
