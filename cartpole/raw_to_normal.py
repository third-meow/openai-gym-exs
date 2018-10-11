import pickle
import numpy as np
from keras.utils import normalize

xtrain = pickle.load(open('saved/raw/xtrain.pickle', 'rb'))
ytrain = pickle.load(open('saved/raw/ytrain.pickle', 'rb'))
xtest = pickle.load(open('saved/raw/xtest.pickle', 'rb'))
ytest = pickle.load(open('saved/raw/ytest.pickle', 'rb'))

new_train = []
new_test = []

def unpack(inp_arr):
    return inp_arr[0]

for train_set in xtrain: 
    new_set = train_set.flatten()
    new_set = normalize(new_set)
    new_train.append(unpack(new_set))

for test_set in xtest:
    new_set = test_set.flatten()
    test_set = normalize(new_set)
    new_test.append(new_set)

new_train = np.array(new_train)
new_test = np.array(new_test)

pickle.dump(new_train, open('saved/normal/xtrain.pickle', 'wb'))
pickle.dump(ytrain, open('saved/normal/ytrain.pickle', 'wb'))
pickle.dump(new_test, open('saved/normal/xtest.pickle', 'wb'))
pickle.dump(ytest, open('saved/normal/ytest.pickle', 'wb'))
