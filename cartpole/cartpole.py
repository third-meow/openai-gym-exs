import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


#load model
mdl = keras.models.load_model('saved/latest_model.h5')

#create enviroment
env = gym.make('CartPole-v0').env
env.reset()

failed = False 

def push_to_prev(prev, new):
    prev[2] = prev[1]
    prev[1] = prev[0]
    prev[0] = new
    return prev


#obs, total_reward, done, ifo = env.step(env.action_space.sample())

#over 100 games
for game_n in range(100):
    total_reward = 0
    obs = env.reset()
    prev_obs = np.array([obs, obs, obs])
    while True:
        #prepare prev_obs for model
        model_input = []
        model_input.append(prev_obs.flatten())
        model_input = keras.utils.normalize(model_input)
        model_input = np.array(model_input)

        #predict step
        nstep = int(round(mdl.predict(model_input)[0][0], 0))

        #take step
        obs, reward, done, _ = env.step(nstep)

        #add reward to total
        total_reward += reward

        #update prev obs
        prev_obs = push_to_prev(prev_obs, obs)

        #if done break
        if done:
            if total_reward < 195:
                #print fail message
                print('Game {} failed with a total reward of {}'.format(game_n, 
                    total_reward))
                #set fail flag
                failed = True
            else:
                print(total_reward)
            break

#if model sucdeeded
if failed == False:
    print("Success!")
    print("Over 100 games, no game failed at less than 195 reward")



#save model
mdl.save('saved/latest_model.h5')
#close enviroment
env.close()
