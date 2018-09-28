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

def unpack(packed):
    return np.argmax(packed)


obs, total_reward, done, ifo = env.step(env.action_space.sample())
obs = np.array(obs[2:])
#env.render()

for i in range(30000):
    while True:
        #get next step from model
        prediction = mdl.predict(np.array([obs]))
        nstep = unpack(prediction)

        #print('{}\n{}-{}'.format(obs, prediction, nstep))

        #make the step, render
        obs, reward, done, ifo = env.step(nstep)
        obs = np.array(obs[2:])
        total_reward += reward
        #env.render()

        #train the model on the outcome of it's previous decision
        if done:
            if nstep == 0:  
                mdl.fit(np.array([obs]), np.array([1]), epochs=7, verbose=0)
            else:
                mdl.fit(np.array([obs]), np.array([0]), epochs=7, verbose=0)

            #if done, reset and break
            print('='*int(total_reward))
            total_reward = 0
            env.reset()
            break

        else:
            mdl.fit(np.array([obs]), np.array([nstep]), epochs=3, verbose=0)
    
        #sleep
        #time.sleep(0.1)


#save model
mdl.save('latest_model.h5')
#close enviroment
env.close()
