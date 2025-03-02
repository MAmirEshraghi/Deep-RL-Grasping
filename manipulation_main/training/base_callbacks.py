import os
import time
import warnings
import logging
from typing import Union, Optional

import gymnasium as gym
import numpy as np

# Update SB3 imports (remove unused sync_envs_normalization)
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback

# If you have an EventCallback class in SB2, in SB3 you may simply use BaseCallback.
# For compatibility, we use BaseCallback here.

class EvalCallback(BaseCallback):
    """
    Callback for evaluating an agent.
    
    :param eval_env: (Union[gym.Env, VecEnv]) The evaluation environment.
    :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger when a new best model is found.
    :param n_eval_episodes: (int) The number of episodes to test the agent.
    :param eval_freq: (int) Frequency (in timesteps) at which to evaluate the agent.
    :param log_path: (str) Path to a folder where evaluations will be saved.
    :param best_model_save_path: (str) Path to a folder where the best model will be saved.
    :param deterministic: (bool) Whether to use deterministic actions during evaluation.
    :param render: (bool) Whether to render the environment during evaluation.
    :param verbose: (int) Verbosity level.
    """
    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1):
        super(EvalCallback, self).__init__(verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render

        # Only wrap eval_env in a DummyVecEnv if it is not already a vectorized environment.
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])
        self.eval_env = eval_env

        self.best_model_save_path = best_model_save_path
        if log_path is not None:
            log_path = os.path.join(log_path, 'evaluations')
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        self.callback = callback_on_new_best

    def _init_callback(self) -> None:
        if type(self.training_env) != type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type: {} != {}"
                          .format(self.training_env, self.eval_env))
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            episode_rewards, episode_lengths = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes,
                render=self.render, deterministic=self.deterministic, return_episode_rewards=True)

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results=self.evaluations_results, ep_lengths=self.evaluations_length)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print("Eval num_timesteps={}, episode_reward={:.2f} +/- {:.2f}".format(
                    self.num_timesteps, mean_reward, std_reward))
                print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

            if mean_reward >= self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self.best_mean_reward = mean_reward
                if self.callback is not None:
                    return self.callback.on_event()
        return True

class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every save_freq steps.
    
    :param save_freq: (int) Frequency to save.
    :param save_path: (str) Path where VecNormalize will be saved as vecnormalize.pkl.
    :param name_prefix: (str) Optional prefix for the saved file.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: Optional[str] = None, verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f'{self.name_prefix}_{self.num_timesteps}_steps.pkl')
            else:
                path = os.path.join(self.save_path, 'vecnormalize.pkl')
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print("Saving VecNormalize to {}".format(path))
        return True

class TrainingTimeCallback(BaseCallback):
    """
    Callback for tracking and logging training time per rollout and per step.
    
    :param verbose: (int) Verbosity level.
    """
    def __init__(self, verbose=0):
        super(TrainingTimeCallback, self).__init__(verbose)
        self.start_time = None
        self.start_tot = None
        self.start_simulator_time = None
        self.time_diffs = np.array([])
        self.sim_time_diffs = np.array([])
        self.tot_time = np.array([])

    def _on_training_start(self) -> None:
        self.start_time = time.process_time()

    def _on_step(self) -> bool:
        if self.start_tot is None:
            self.start_tot = time.process_time()
        else:
            self.tot_time = np.append(time.process_time() - self.start_tot, self.tot_time)
            self.start_tot = None
        if self.start_simulator_time:
            time_diff = time.process_time() - self.start_simulator_time
            self.sim_time_diffs = np.append(self.sim_time_diffs, time_diff)
        if len(self.sim_time_diffs) >= 1000:
            logging.info("Average time per env step: {}".format(np.mean(self.sim_time_diffs)))
            self.sim_time_diffs = np.array([])
        if len(self.tot_time) >= 1000:
            logging.info("Average total time per step: {}".format(np.mean(self.tot_time)))
            self.tot_time = np.array([])
        return True

    def _on_rollout_start(self) -> None:
        if self.num_timesteps == 0:
            return
        else:
            end_time = time.process_time()
            time_diff = end_time - self.start_time
            self.time_diffs = np.append(self.time_diffs, time_diff)
            if self.num_timesteps % 1000 == 0:
                logging.info("Average training step time: {}".format(np.mean(self.time_diffs)))
                self.time_diffs = np.array([])
            self.start_time = end_time
        self.start_simulator_time = time.process_time()

    def _on_rollout_end(self) -> None:
        self.start_time = time.process_time()

