import random
import numpy as np


# # # # # # # # # #
#   Actions         
#       0 = left    
#       1 = right   
# # # # # # # # # #

def create_train_data():
    xy = []
    for i in [v / 100.0 for v in range(25, 1200, 25)]:
        for j in [e / 100.0 for e in range(25, 10000, 25)]:
            xy.append([np.array([i, j]), 1])


    for i in [v / 100.0 for v in range(-25, -1200, -25)]:
        for j in [e / 100.0 for e in range(-25, -10000, -25)]:
            xy.append([np.array([i, j]), 0])

    #print(xy[:3], end='\n\n')
    random.shuffle(xy)
    #print(xy[:3])

    x = []
    y = []
    for i in xy:
        x.append(i[0])
        y.append(i[1])

    x = np.array(x)
    y = np.array(y)

    split = int(len(x) / 10)
    xtrain, xtest = x[split:], x[:split]
    ytrain, ytest = y[split:], y[:split]

    return xtrain, ytrain, xtest, ytest

