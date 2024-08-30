import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.utils import check_env
import assistive_gym

num_processes = multiprocessing.cpu_count()/2
# num_processes = 1
seed = 0
env_name = "assistive_gym:WipingEnv-v0"

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Load your environment
env = gym.make(env_name)
# Check the environment
check_env(env)

# Configuration for PPO
config = DEFAULT_CONFIG.copy()
config['train_batch_size'] = 5000
config['num_sgd_iter'] = 50
config['sgd_minibatch_size'] = 128
config['lambda'] = 0.95
config['model']['fcnet_hiddens'] = [100, 100]
config['num_workers'] = num_processes
config['num_cpus_per_worker'] = 0
config['seed'] = seed
config['log_level'] = 'ERROR'

# Specify the directory to save the trained model
checkpoint_path = "./assistive_gym/trained_models"
config["output"] = checkpoint_path
config["output_compress_columns"] = []  # Avoid compressing for easier recovery

# Build a PPOTrainer object from the config and run 1 training iteration.
agent = PPOTrainer(config, env_name)
result = agent.train()
print("Training iteration results:", result)

# Save the model
checkpoint = agent.save(checkpoint_path)
print("Checkpoint saved at:", checkpoint)

# Shutdown Ray when done
ray.shutdown()
