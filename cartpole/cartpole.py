import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


#load model
mdl = keras.models.load_model('latest_model.h5')

#create enviroment
env = gym.make('CartPole-v0')
env.reset()

fail_n = 0 
upcycles = 0

obs, total_reward, done, ifo = env.step(env.action_space.sample())
obs = np.array([obs])
env.render()


def unpack(packed):
    return int(packed[0][0])
for i in range(1000):
    while True:
        #get next step from model
        prediction = mdl.predict(obs)
        nstep = unpack(prediction)

        #make the step, render
        obs, reward, done, ifo = env.step(nstep)
        obs = np.array([obs])
        total_reward += reward
        env.render()

        #train the model on the outcome of it's previous decision
        if done:
            if nstep == 0:  
                mdl.fit([obs], [1], epochs=7, verbose=0)
            else:
                mdl.fit([obs], [0], epochs=7, verbose=0)

            #if done, reset and break
            print(total_reward)
            upcycles = 0
            total_reward = 0
            env.reset()
            break

        else:
            mdl.fit(obs, prediction, epochs=1, verbose=0)
            upcycles += 1
    
        #sleep
        time.sleep(0.1)
    


#save model
mdl.save('latest_model.h5')
#close enviroment
env.close()
