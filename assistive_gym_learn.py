import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
import time
from ray.rllib.agents import ppo
import assistive_gym
import matplotlib.pyplot as plt


def setup_config(env, algo, render, coop=False, seed=0, extra_configs={}):
    # num_processes = multiprocessing.cpu_count()
    if render:
        num_processes = 1
    else:
        num_processes = max(1, int(multiprocessing.cpu_count() / 2))  # Use half of your cores

    print(f'num_processes: {num_processes}')
    if algo == 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        config['train_batch_size'] = 19200
        config['num_sgd_iter'] = 50
        config['sgd_minibatch_size'] = 128
        config['lambda'] = 0.95
        config['model']['fcnet_hiddens'] = [100, 100]
    else:
        raise ValueError('invalid algorithm')
    
    config['num_workers'] = num_processes
    config['num_cpus_per_worker'] = 0
    config['seed'] = seed
    config['log_level'] = 'ERROR'

    return {**config, **extra_configs}

def load_policy(env, algo, env_name, policy_path=None, coop=False, seed=0, extra_configs={}, render=False):
    if algo == 'ppo':
        agent = ppo.PPOTrainer(setup_config(env, algo, render, coop, seed, extra_configs), env_name)
    else:
        raise ValueError('invalid algorithm')
    
    if policy_path != '':
        if 'checkpoint' in policy_path:
            agent.restore(policy_path)
        else:
            # Find the most recent policy in the directory
            directory = os.path.join(policy_path, algo, env_name)
            files = [f.split('_')[-1] for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
            files_ints = [int(f) for f in files]
            if files:
                checkpoint_max = max(files_ints)
                checkpoint_num = files_ints.index(checkpoint_max)
                checkpoint_path = os.path.join(directory, 'checkpoint_%s' % files[checkpoint_num], 'checkpoint-%d' % checkpoint_max)
                print(f'checkpoint_path: {checkpoint_path}')
                agent.restore(checkpoint_path)
                return agent, checkpoint_path
            return agent, None
    return agent, None

def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make(env_name)
    else:
        raise ValueError('coop is not supported')
    
    env.seed(seed)
    return env

def train(env_name, algo, timesteps_total=1000000, save_dir='./trained_models/', load_policy_path='', coop=False, seed=0, extra_configs={}):
    render = False
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env(env_name)
    agent, checkpoint_path = load_policy(env, algo, env_name, load_policy_path, coop, seed, extra_configs, render)
    env.close()

    timesteps = 0
    start_time = time.time()

    while timesteps < timesteps_total:
        result = agent.train()
        timesteps = result['timesteps_total']

        print(f"Iteration: {result['training_iteration']}, total timesteps: {result['timesteps_total']}, total time: {result['time_total_s']:.1f}, FPS: {result['timesteps_total']/result['time_total_s']:.1f}, mean reward: {result['episode_reward_mean']:.1f}, min/max reward: {result['episode_reward_min']:.1f}/{result['episode_reward_max']:.1f}")
        sys.stdout.flush()

        # Delete the old saved policy
        if checkpoint_path is not None:
            shutil.rmtree(os.path.dirname(checkpoint_path), ignore_errors=True)
        # Save the recently trained policy
        checkpoint_path = agent.save(os.path.join(save_dir, algo, env_name))

    training_time = time.time() - start_time
    print(f"training time: {training_time} s")

    return checkpoint_path

def render_policy(env, env_name, algo, policy_path, coop=False, colab=False, seed=0, n_episodes=1, extra_configs={}):
    render = True
    ray.init(num_cpus=1, ignore_reinit_error=True, log_to_driver=False)
    if env is None:
        env = make_env(env_name, seed=seed)
    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs, render)

    if not colab:
        env.render()

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            if coop:
                raise ValueError('coop not implemented')
            else:
                # Compute the next action using the trained policy
                action = test_agent.compute_action(obs)
                # Step the simulation forward using the action from our trained policy
                obs, reward, done, info = env.step(action)

                number_of_targets_cleared = info['number_of_targets_cleared']
                task_percentage = info['task_percentage']
                total_target_count = info['total_target_count']
                feasible_target_count = info['feasible_target_count']
        
            print(f'number_of_targets_cleared: {number_of_targets_cleared}/{feasible_target_count}, task_percentage: {task_percentage}, reward: {reward}')
    
    env.close()


def evaluate_policy(env_name, algo, policy_path, n_episodes=100, coop=False, seed=0, verbose=False, extra_configs={}):
    render = False
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env(env_name, coop, seed=seed)
    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs, render)

    rewards = []
    task_successes = []
    number_of_targets_cleared_list = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        reward_total = 0.0
        task_success = 0.0
        while not done:
            if coop:
                raise ValueError('coop not implemented')
            else:
                action = test_agent.compute_action(obs)
                obs, reward, done, info = env.step(action)
            reward_total += reward
            task_success = info['task_success']
            number_of_targets_cleared = info['number_of_targets_cleared']

        rewards.append(reward_total)
        task_successes.append(task_success)
        number_of_targets_cleared_list.append(number_of_targets_cleared)
        if verbose:
            print('Reward total: %.2f, task success: %r, number of targets cleared %r' % (reward_total, task_success, number_of_targets_cleared))
        sys.stdout.flush()
    env.close()

    print('\n', '-'*50, '\n')
    # print('Rewards:', rewards)
    print('Reward Mean:', np.mean(rewards))
    print('Reward Std:', np.std(rewards))

    # print('Task Successes:', task_successes)
    print('Task Success Mean:', np.mean(task_successes))
    print('Task Success Std:', np.std(task_successes))

    print('Number of Targets Cleared Mean:', np.mean(number_of_targets_cleared_list))
    print('Number of Targets Cleared Std:', np.std(number_of_targets_cleared_list))
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('--env', default='assistive_gym:WipingEnv-v0',
                        help='Environment to train on (default: WipingEnv-v0)')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train a new policy')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate a trained policy over n_episodes')
    parser.add_argument('--train-timesteps', type=int, default=1000000,
                        help='Number of simulation timesteps to train a policy (default: 1000000)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--load-policy-path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    parser.add_argument('--render-episodes', type=int, default=1,
                        help='Number of rendering episodes (default: 1)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--colab', action='store_true', default=False,
                        help='Whether rendering should generate an animated png rather than open a window (e.g. when using Google Colab)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to output more verbose prints')
    args = parser.parse_args()

    coop = False
    checkpoint_path = None

    if args.train:
        checkpoint_path = train(args.env, args.algo, timesteps_total=args.train_timesteps, save_dir=args.save_dir, load_policy_path=args.load_policy_path, coop=coop, seed=args.seed)
    if args.render:
        render_policy(None, args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, coop=coop, colab=args.colab, seed=args.seed, n_episodes=args.render_episodes)
    if args.evaluate:
        evaluate_policy(args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, n_episodes=args.eval_episodes, coop=coop, seed=args.seed, verbose=args.verbose)
