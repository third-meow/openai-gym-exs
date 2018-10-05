import gym
import random
import pickle
import numpy as np

def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

def create_train_data():
    train_data = []
    
    env = gym.make('CartPole-v0')
    env.reset()

    while len(train_data) <= 360000:
        game_data = []
        #setup game
        obs = env.reset()
        prev_obs = obs
        total_reward = 0 

        #run random game, do random moves; record moves
        while True:
            nstep = env.action_space.sample()
            obs, reward, done, ifo = env.step(nstep)
            total_reward += reward

            game_data.append([prev_obs, nstep])
            prev_obs = obs

            if done:
                #if game went well
                if total_reward > 100:
                    # append all the 'good' moves from game data 
                    # (except last 10, they might have lead to the fail)
                    # to train_data
                    for good_move in game_data[:-10]:
                        train_data.append(good_move)
                break

    x = []
    y = []
    for i in train_data:
        x.append(i[0])
        y.append(i[1])

    x = np.array(x)
    y = np.array(y)

    #scale down data
    for obs in x:
        obs[0] = obs[0]/4.8
        obs[1] = sigmoid(obs[1])
        obs[2] = obs[2]/4.2
        obs[3] = sigmoid(obs[3])

    split = int(len(x) / 10)
    xtrain, xtest = x[split:], x[:split]
    ytrain, ytest = y[split:], y[:split]

    return xtrain, ytrain, xtest, ytest

if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = create_train_data()
    pickle.dump(xtrain, open('saved/xtrain.pickle', 'wb'))
    pickle.dump(ytrain, open('saved/ytrain.pickle', 'wb'))
    pickle.dump(xtest, open('saved/xtest.pickle', 'wb'))
    pickle.dump(ytest, open('saved/ytest.pickle', 'wb'))
