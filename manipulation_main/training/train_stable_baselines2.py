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
for robotic grasping. It parses command-line arguments to either train a new model (using algorithms
such as SAC, DQN, BDQ, etc.) or run a previously trained model. The script sets up the simulation environment,
loads configuration files, and coordinates with helper modules (e.g., sb_helper, io_utils, TimeFeatureWrapper,
and run_agent) to execute the desired operation.
"""

import argparse
import logging
import os
import gym

import numpy as np
import stable_baselines as sb
#sb_helper: Create an SBPolicy object, which encapsulates the setup and learning procedure for the RL algorithm. This helper abstracts away many details of initializing and training the model.
import sb_helper #for setting up and managing the training process.
import tensorflow as tf
import manipulation_main

# Import specific RL algorithm classes and utilities.
from stable_baselines import SAC
from stable_baselines.common.callbacks import BaseCallback, EvalCallback
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.deepq.policies import MlpPolicy as DQNMlpPolicy
from stable_baselines.sac.policies import MlpPolicy as sacMLP
from stable_baselines.bench import Monitor

#TimeFeatureWrapper: Imported and conditionally applied to the Gym environment if the --timefeature flag is set. It wraps the environment to add extra time-related features
from manipulation_main.training.wrapper import TimeFeatureWrapper # for optionally enhancing the environment. 
#in-utils: 
from manipulation_main.common import io_utils #for configuration loading and saving.
#run_agent: Used in the run(args) function to execute the trained agent in the environment. It handles the agent–environment interaction loop during evaluation.
from manipulation_main.utils import run_agent #for executing a trained agent during evaluation.

# Suppress TensorFlow logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def train(args):
    # Load configuration from YAML file specified in the command line argument
    config = io_utils.load_yaml(args.config)
    
    # Create directory to save the model and logs
    os.mkdir(args.model_dir)
    # Create a subdirectory for saving the best model
    os.mkdir(args.model_dir + "/best_model")
    model_dir = args.model_dir
    algo = args.algo

    # Update simulation settings based on provided arguments
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
    
    # Set the save directory for the algorithm's configuration
    config[algo]['save_dir'] = model_dir

    # Create the training environment. Use TimeFeatureWrapper if requested.
    if args.timefeature:
        env = DummyVecEnv([lambda: TimeFeatureWrapper(gym.make('gripper-env-v0', config=config))])
    else:
        env = DummyVecEnv([lambda: Monitor(gym.make('gripper-env-v0', config=config), os.path.join(model_dir, "log_file"))])
    
    # Prepare a separate evaluation configuration (no real-time simulation or visualization)
    config["algorithm"] = args.algo.lower()
    config_eval = config
    config_eval['simulation']['real_time'] = False
    config_eval['simulation']['visualize'] = False

    # Save configuration for reproducibility
    io_utils.save_yaml(config, os.path.join(args.model_dir, 'config.yaml'))
    io_utils.save_yaml(config, os.path.join(args.model_dir, 'best_model/config.yaml'))

    # Create the test environment with evaluation flags
    if args.timefeature:
        test_env = DummyVecEnv([lambda: TimeFeatureWrapper(gym.make('gripper-env-v0', config=config_eval, evaluate=True, validate=True))])
    else:
        test_env = DummyVecEnv([lambda: gym.make('gripper-env-v0', config=config_eval, evaluate=True, validate=True)])
    
    # Initialize the Stable Baselines helper with training and test environments
    sb_help = sb_helper.SBPolicy(env, test_env, config, model_dir, args.load_dir, algo)
    # Begin training
    sb_help.learn()
    # Close environments after training
    env.close()
    test_env.close()

def run(args):
    # Extract the directory path from the model file path to locate the configuration file
    top_folder_idx = args.model.rfind('/')
    top_folder_str = args.model[0:top_folder_idx]
    config_file = top_folder_str + '/config.yaml'
    config = io_utils.load_yaml(config_file)
    normalize = config.get("normalize", False)

    # Update simulation settings for visualization if requested
    if args.visualize:
        config['simulation']['real_time'] = False
        config['simulation']['visualize'] = True

    # Create a test environment for running the agent
    task = DummyVecEnv([lambda: gym.make('gripper-env-v0', config=config, evaluate=True, test=args.test)])

    # If normalization was applied during training, load the normalization parameters
    if normalize:
        task = VecNormalize(task, training=False, norm_obs=True, norm_reward=True, clip_obs=10.)
        task = VecNormalize.load(os.path.join(top_folder_str, 'vecnormalize.pkl'), task)
        
    # Determine and load the correct algorithm's trained model
    model_lower = args.model.lower() 
    if 'trpo' == config["algorithm"]: 
        agent = sb.TRPO.load(args.model)
    elif 'sac' == config["algorithm"]:
        agent = sb.SAC.load(args.model)
    elif 'ppo' == config["algorithm"]:
        agent = sb.PPO2.load(args.model)
    elif 'dqn' == config["algorithm"]:
        agent = sb.DQN.load(args.model)
    elif 'bdq' == config["algorithm"]:
        agent = sb.BDQ.load(args.model)
    else:
        raise Exception("Algorithm not supported.")
    
    print("Run the agent")
    # Run the agent in the environment; args.stochastic controls action randomness
    run_agent(task, agent, args.stochastic)
    task.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Sub-command for training a policy
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    train_parser.add_argument('--algo', type=str, required=True, help="RL algorithm to use (e.g., SAC, DQN, BDQ).")
    train_parser.add_argument('--model_dir', type=str, required=True, help="Directory where models and logs will be saved.")
    train_parser.add_argument('--load_dir', type=str, help="Optional: Directory to load an existing model.")

    train_parser.add_argument('--timestep', type=str, help="Total training timesteps.")
    train_parser.add_argument('-s', '--simple', action='store_true', help="Use a simplified environment.")
    train_parser.add_argument('-sh', '--shaped', action='store_true', help="Use a shaped reward function.")
    train_parser.add_argument('-v', '--visualize', action='store_true', help="Enable visualization during simulation.")
    train_parser.add_argument('-tf', '--timefeature', action='store_true', help="Add time feature to the environment.")

    train_parser.set_defaults(func=train)

    # Sub-command for running a trained policy
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--model', type=str, required=True, help="Path to the trained model file.")
    run_parser.add_argument('-v', '--visualize', action='store_true', help="Enable visualization during execution.")
    run_parser.add_argument('-t', '--test', action='store_true', help="Run on the test dataset.")
    run_parser.add_argument('-s', '--stochastic', action='store_true', help="Use stochastic actions during execution.")
    run_parser.set_defaults(func=run)

    logging.getLogger().setLevel(logging.DEBUG)

    # Parse arguments and call the appropriate function (train or run)
    args = parser.parse_args()
    args.func(args)

"""
Necessary Commands to Run the Code
To Train a Model
An example command to train an agent using the SAC algorithm with visualization and a defined number of timesteps:



python manipulation_main/training/train_stable_baselines.py train --config config/gripper_grasp.yaml --algo SAC --model_dir trained_models/SAC_full --timestep 100000 -v
Explanation:
train: Calls the training sub-command.
--config config/gripper_grasp.yaml: Loads the configuration from the specified YAML file.
--algo SAC: Uses the SAC algorithm.
--model_dir trained_models/SAC_full: Saves models and logs to this directory.
--timestep 100000: Trains for 100,000 timesteps.
-v: Enables visualization during training.
To Run a Trained Model
An example command to run (evaluate) a pre-trained model:


python manipulation_main/training/train_stable_baselines.py run --model trained_models/SAC_full/best_model/best_model.zip -v -t -s
Explanation:
run: Calls the run sub-command.
--model trained_models/SAC_full/best_model/best_model.zip: Specifies the path to the pre-trained model.
-v: Enables visualization during the run.
-t: Indicates that the test dataset should be used.
-s: Runs the agent in stochastic mode (if applicable).

"""