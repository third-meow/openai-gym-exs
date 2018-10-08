import gym
import random
import pickle
import numpy as np

def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

def push_to_prev(prev, new):
    prev[2] = prev[1]
    prev[1] = prev[0]
    prev[0] = new
    return prev

def create_train_data():
    #holds main training data
    train_data = []
    
    #create enviroment
    env = gym.make('CartPole-v0')

    while len(train_data) <= 360000:
        #data on obs and move for just this game, 
        #might be included in train data later
        game_data = []

        #setup game
        obs = env.reset()

        #keep record of prev 3 obs-move pairs
        prev_obs = []
        prev_obs.append(obs)
        prev_obs.append(obs)
        prev_obs.append(obs)
        total_reward = 0 

        #run random game, do random moves; record moves
        while True:
            #generate random move
            nstep = env.action_space.sample()

            #do random move
            obs, reward, done, ifo = env.step(nstep)
            
            #tally reward
            total_reward += reward

            #append last 3 obs and current move to game data
            game_data.append([prev_obs[:], nstep])
            #update last 3 obs
            prev_obs = push_to_prev(prev_obs, obs)

            #if game has terminated
            if done:
                #if game went well
                if total_reward > 100:
                    # append all the 'good' moves from game data 
                    # (except last and first 10, they are not as reliable)
                    # to train_data
                    for good_move in game_data[10:-10]:
                        train_data.append(good_move)
                #then break from game loop
                break

    #split train data into x and y
    x = []
    y = []
    for i in train_data:
        x.append(i[0])
        y.append(i[1])

    #turn x and y into numpy array
    x = np.array(x)
    y = np.array(y)


    #split data into training and testing data
    split = int(len(x) / 10)
    xtrain, xtest = x[split:], x[:split]
    ytrain, ytest = y[split:], y[:split]

    return xtrain, ytrain, xtest, ytest

if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = create_train_data()
    pickle.dump(xtrain, open('saved/raw/xtrain.pickle', 'wb'))
    pickle.dump(ytrain, open('saved/raw/ytrain.pickle', 'wb'))
    pickle.dump(xtest, open('saved/raw/xtest.pickle', 'wb'))
    pickle.dump(ytest, open('saved/raw/ytest.pickle', 'wb'))
