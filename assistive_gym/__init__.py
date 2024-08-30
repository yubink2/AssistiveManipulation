"""Register new gym environment"""
from gym.envs.registration import register

register(
    id='WipingEnv-v0',
    entry_point='assistive_gym.envs:WipingEnv',
    max_episode_steps=150,
)
