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
            xy.append([np.array([0, 0, i, j]), 1])


    for i in [v / 100.0 for v in range(-25, -1200, -25)]:
        for j in [e / 100.0 for e in range(-25, -10000, -25)]:
            xy.append([np.array([0, 0, i, j]), 0])

    random.shuffle(xy)

    x = []
    y = []
    for i in xy:
        x.append(i[0])
        y.append(i[1])


    print(y)

create_train_data()

    
