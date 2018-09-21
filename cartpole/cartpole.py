import gym
import time
import numpy as np

env = gym.make('CartPole-v0')
env.reset()
env.render()

for i in range(99):
    print(i)
    obs, reward, done, ifo = env.step(env.action_space.sample())
    if done:
        print("failed?")
        env.reset()
    env.render()
    time.sleep(0.1)

env.close()
