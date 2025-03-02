"""
Usage Examples:
---------------
To Train a Model (using SAC algorithm, 100k timesteps, with visualization enabled):
    python manipulation_main/training/train_stable_baselines.py train --config config/gripper_grasp.yaml --algo SAC --model_dir trained_models/SAC_full --timestep 100000 -v

To Run a Trained Model (using the saved best model, with visualization, on test dataset, stochastic mode):
    python manipulation_main/training/train_stable_baselines.py run --model trained_models/SAC_full/best_model/best_model.zip -v -t -s
"""
"""
This script serves as the main entry point for training and evaluating reinforcement learning agents
for robotic grasping using Stable-Baselines3. It parses command-line arguments to either train a new model
(using algorithms such as SAC, PPO, DQN, TD3, or A2C) or run a previously trained model. The script sets up the
simulation environment, loads configuration files, and coordinates with helper modules (e.g., sb_helper, io_utils,
TimeFeatureWrapper, and run_agent) to execute the desired operation.
"""

import argparse
import logging
import os

# Use gymnasium for compatibility with Stable-Baselines3.
import gymnasium as gym

import numpy as np
# Import Stable-Baselines3 (which is built on PyTorch)
import stable_baselines3 as sb3
import sb_helper  # Custom helper to set up and manage the training process (updated for SB3).
import manipulation_main

# Import specific algorithms and utilities from Stable-Baselines3.
from stable_baselines3 import SAC, PPO, DQN, TD3, A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# TimeFeatureWrapper: Optionally adds a time feature to the environment's observations.
from manipulation_main.training.wrapper import TimeFeatureWrapper
# io_utils: For configuration loading and saving.
from manipulation_main.common import io_utils
# run_agent: For executing a trained agent during evaluation.
from manipulation_main.utils import run_agent

# Suppress extraneous logging (if needed).
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train(args):
    # Load configuration from YAML file.
    config = io_utils.load_yaml(args.config)
    
    # Create directory to save the model and logs.
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.join(args.model_dir, "best_model"), exist_ok=True)
    model_dir = args.model_dir
    algo = args.algo

    # Update simulation settings based on provided command-line arguments.
    if args.visualize:
        config['simulation']['real_time'] = False
        config['simulation']['visualize'] = True
    if args.simple:
        logging.info("Simplified environment is set")
        config['simplified'] = True
    if args.shaped:
        logging.info("Shaped reward function is being used")
        config['reward']['shaped'] = True
    if args.timestep:
        config[algo]['total_timesteps'] = args.timestep
    if not args.algo == 'DQN':
        config['robot']['discrete'] = False    
    else:
        config['robot']['discrete'] = True
    
    # Set the save directory in the configuration.
    config[algo]['save_dir'] = model_dir

    # Create the training environment; use TimeFeatureWrapper if requested.
    if args.timefeature:
        env = DummyVecEnv([lambda: TimeFeatureWrapper(gym.make('gripper-env-v0', config=config))])
    else:
        env = DummyVecEnv([lambda: Monitor(gym.make('gripper-env-v0', config=config), os.path.join(model_dir, "log_file"))])
    
    # Prepare an evaluation configuration that disables real-time simulation and visualization.
    config["algorithm"] = args.algo.lower()
    config_eval = config
    config_eval['simulation']['real_time'] = False
    config_eval['simulation']['visualize'] = False

    # Save configuration for reproducibility.
    io_utils.save_yaml(config, os.path.join(args.model_dir, 'config.yaml'))
    io_utils.save_yaml(config, os.path.join(args.model_dir, 'best_model/config.yaml'))

    # Create the test environment with evaluation flags.
    if args.timefeature:
        test_env = DummyVecEnv([lambda: TimeFeatureWrapper(gym.make('gripper-env-v0', config=config_eval, evaluate=True, validate=True))])
    else:
        test_env = DummyVecEnv([lambda: gym.make('gripper-env-v0', config=config_eval, evaluate=True, validate=True)])
    
    # Initialize the Stable-Baselines3 helper with training and test environments.
    sb_help = sb_helper.SBPolicy(env, test_env, config, model_dir, args.load_dir, algo)
    # Begin training.
    sb_help.learn()
    # Close environments after training.
    env.close()
    test_env.close()

def run(args):
    # Extract the top-level directory from the model file path to load the configuration.
    top_folder_idx = args.model.rfind('/')
    top_folder_str = args.model[0:top_folder_idx]
    config_file = os.path.join(top_folder_str, 'config.yaml')
    config = io_utils.load_yaml(config_file)
    normalize = config.get("normalize", False)

    # Update simulation settings for visualization if requested.
    if args.visualize:
        config['simulation']['real_time'] = False
        config['simulation']['visualize'] = True

    # Create a test environment for running the agent.
    task = DummyVecEnv([lambda: gym.make('gripper-env-v0', config=config, evaluate=True, test=args.test)])
    
    # Load normalization parameters if they were used during training.
    if normalize:
        task = VecNormalize(task, training=False, norm_obs=True, norm_reward=True, clip_obs=10.)
        task = VecNormalize.load(os.path.join(top_folder_str, 'vecnormalize.pkl'), task)
        
    # Determine and load the correct algorithm's trained model.
    algo_lower = config["algorithm"].lower() 
    if algo_lower == 'sac':
        agent = SAC.load(args.model, env=task)
    elif algo_lower == 'ppo':
        agent = PPO.load(args.model, env=task)
    elif algo_lower == 'dqn':
        agent = DQN.load(args.model, env=task)
    elif algo_lower == 'td3':
        agent = TD3.load(args.model, env=task)
    elif algo_lower == 'a2c':
        agent = A2C.load(args.model, env=task)
    else:
        raise Exception("Algorithm not supported in Stable-Baselines3.")
    
    print("Run the agent")
    # Run the agent in the environment; the --stochastic flag controls action randomness.
    run_agent(task, agent, args.stochastic)
    task.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Sub-command for training a policy.
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    train_parser.add_argument('--algo', type=str, required=True, help="RL algorithm to use (e.g., SAC, PPO, DQN, TD3, A2C).")
    train_parser.add_argument('--model_dir', type=str, required=True, help="Directory where models and logs will be saved.")
    train_parser.add_argument('--load_dir', type=str, help="Optional: Directory to load an existing model.")
    train_parser.add_argument('--timestep', type=str, help="Total training timesteps.")
    train_parser.add_argument('-s', '--simple', action='store_true', help="Use a simplified environment.")
    train_parser.add_argument('-sh', '--shaped', action='store_true', help="Use a shaped reward function.")
    train_parser.add_argument('-v', '--visualize', action='store_true', help="Enable visualization during simulation.")
    train_parser.add_argument('-tf', '--timefeature', action='store_true', help="Add time feature to the environment.")
    train_parser.set_defaults(func=train)

    # Sub-command for running a trained policy.
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--model', type=str, required=True, help="Path to the trained model file.")
    run_parser.add_argument('-v', '--visualize', action='store_true', help="Enable visualization during execution.")
    run_parser.add_argument('-t', '--test', action='store_true', help="Run on the test dataset.")
    run_parser.add_argument('-s', '--stochastic', action='store_true', help="Use stochastic actions during execution.")
    run_parser.set_defaults(func=run)

    logging.getLogger().setLevel(logging.DEBUG)

    args = parser.parse_args()
    args.func(args)

"""
Necessary Commands to Run the Code
To Train a Model:
    python manipulation_main/training/train_stable_baselines.py train --config config/gripper_grasp.yaml --algo SAC --model_dir trained_models/SAC_full --timestep 100000 -v
To Run a Trained Model:
    python manipulation_main/training/train_stable_baselines.py run --model trained_models/SAC_full/best_model/best_model.zip -v -t -s
"""

