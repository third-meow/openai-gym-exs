import gym
import numpy as np

env = gym.make('Taxi-v2')

qtab = np..zeros([env.obervation_space.n, env.action_space.n])
print(qtab)
