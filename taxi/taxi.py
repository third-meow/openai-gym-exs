import random
#from time import sleep
import gym
import numpy as np

env = gym.make('Taxi-v2').env

qtab = np.zeros([env.observation_space.n, env.action_space.n])

explore_chance = 0.2
explore_dropoff = 0.00001
learnrate = 0.1
gamma = 0.6


total_reward  = 0
state = env.reset()



def invert(x):
    return (1.0 - x)

for i in range(1000):
    #explore_chance += (500*explore_dropoff)
    while True:

        if random.uniform(0, 1) < explore_chance:
            nstep = env.action_space.sample()
            #explore_chance -= explore_dropoff
        else:
            nstep = np.argmax(qtab[state])


        prev_state = state
        state, reward, done, ifo = env.step(nstep)

        #update total reward
        total_reward += reward
        
        
        #update qtab
        current = qtab[prev_state][nstep]
        next_step_best = np.max(qtab[state])
        qtab[prev_state][nstep] = (
            (invert(learnrate) * current) 
            + (learnrate * (reward + (gamma * next_step_best)))
        )




        if done:
            print('''It: {}'s reward was: {}'''.format(i, total_reward))
            break

        
