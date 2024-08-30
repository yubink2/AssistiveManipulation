import gym, sys, argparse
import numpy as np
import assistive_gym


env = gym.make("assistive_gym:WipingEnv-v0")
env.render()
observation = env.reset()
done = False

for _ in range(500):
    for _ in range(10):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, done, info = env.step(action)
        
        number_of_targets_cleared = info['number_of_targets_cleared']
        task_success = info['task_success']
        # print(observation)
        # print(f'reward: {reward}, number_of_targets_cleared: {number_of_targets_cleared}, task_success: {task_success}')

        if done:
            observation = env.reset()
    
    observation = env.reset()
    done = False

env.close()