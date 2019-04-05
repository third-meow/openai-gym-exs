import random
import gym

env = gym.make('Acrobot-v1').env
env.reset()

#
# So, I dont really know what I'm doing here but, here is the idea:
#
# actions_tab holds data such:
# actions_tab : [[prev-obs, action, resulting-reward],...]
# using explore_chance agent decides to 
#     A) take random action, if reward is good, store in actions_tab
#  or B) find closest 'good' action from actions_tab using KNN (https://bit.ly/YVTLBc)
#
# overtime explore_chance decreases so agent takes less random actions
#

# stores number of steps taken
step_count = 0
actions_tab = []
explore_chance = 0.5
prev_obs, _, _, _ = enc.step(env.action_space.sample())

while True:
    step_count += 1

    if random.uniform() < explore_chance:
        # take random action
        nstep = env.action_space.sample()
        obs, rwd, done, _ = env.step(nstep)

        if (rwd
        tab.append([prev_obs, nstep, rwd])

    else:
        # take action from action_tab
        nstep = env.action

    env.render()
    if done:
        print(f'Done, in {step_count} steps')
        break

    prev_obs = obs
