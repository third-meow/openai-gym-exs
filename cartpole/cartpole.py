import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


#load model
mdl = keras.models.load_model('saved/latest_model.h5')

#create enviroment
env = gym.make('CartPole-v0')
env.reset()

fail_n = 0 

def push_to_prev(prev, new):
    prev[2] = prev[1]
    prev[1] = prev[0]
    prev[0] = new
    return prev

def unpack(packed):
    return np.argmax(packed)

def predict(in_data):
    keras.utils.normalize(in_data)


obs, total_reward, done, ifo = env.step(env.action_space.sample())
#env.render()

for _ in range(100):
    obs = env.reset()
    prev_obs = np.empty([3])
    prev_obs = np.append(prev_obs, [obs])
    prev_obs = np.append(prev_obs, [obs])
    prev_obs = np.append(prev_obs, [obs])
    while True:
        print(prev_obs)
        print(prev_obs.shape)
        #predict step
        nstep = mdl.predict(prev_obs)
        print(nstep)
        #take step
        #record obs
        #update prev obs
        #if done break
        break


#save model
mdl.save('saved/latest_model.h5')
#close enviroment
env.close()
