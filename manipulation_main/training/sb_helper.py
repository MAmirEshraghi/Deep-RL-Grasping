import os
import time
import logging
import numpy as np
import stable_baselines3 as sb3
import custom_obs_policy

from base_callbacks import EvalCallback, TrainingTimeCallback, SaveVecNormalizeCallback

# For SAC, we import policies from the SAC submodule:
from stable_baselines3.sac.policies import MlpPolicy as sacMlp, CnnPolicy as sacCnn

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional values to tensorboard using SB3's logger.
    Logs the success rate of the agent at a specified frequency.
    """
    def __init__(self, task, log_freq, model_name, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.task = task
        self.model_name = model_name
        self.log_freq = log_freq
        self.old_timestep = -1

    def _on_step(self) -> bool:
        history = self.task.get_attr("history")[0]
        sr = self.task.get_attr("sr_mean")[0]
        curr = self.task.get_attr("curriculum")[0]
        if len(history) != 0 and self.num_timesteps != self.old_timestep:
            if self.num_timesteps % self.log_freq == 0:
                logging.info("model: {} Success Rate: {} Timestep Num: {} lambda: {}".format(
                    self.model_name, sr, self.num_timesteps, curr._lambda))
            self.logger.record('success_rate', sr)
            self.old_timestep = self.num_timesteps
        return True

class SBPolicy:
    def __init__(self, env, test_env, config, model_dir, load_dir=None, algo='SAC', log_freq=1000):
        self.env = env
        self.test_env = test_env
        self.algo = algo
        self.config = config
        self.load_dir = load_dir
        self.model_dir = model_dir
        self.log_freq = log_freq
        self.norm = config['normalize']
 
    def learn(self):
        # Use deterministic actions for evaluation.
        eval_path = os.path.join(self.model_dir, "best_model")
        save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=eval_path)
        if self.norm:
            self.test_env = VecNormalize(self.test_env, norm_obs=True, norm_reward=False, clip_obs=10.)
        eval_callback = EvalCallback(
            self.test_env,
            best_model_save_path=eval_path,
            log_path=os.path.join(eval_path, 'logs'),
            eval_freq=50000,
            n_eval_episodes=10,
            callback_on_new_best=save_vec_normalize,
            deterministic=True,
            render=False
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=25000,
            save_path=os.path.join(self.model_dir, 'logs'),
            name_prefix='rl_model'
        )
        time_callback = TrainingTimeCallback()
        tensorboard_file = (None if self.config[self.algo]['tensorboard_logs'] is None 
                           else os.path.join("tensorboard_logs", self.model_dir))
        
        if self.algo.upper() == 'SAC':
            # Unwrap the environment to access custom attributes.
            env0 = self.env.envs[0].unwrapped
            if not env0.is_simplified() and (env0.depth_obs or env0.full_obs):
                # Use our custom features extractor (returns keys: features_extractor_class and features_extractor_kwargs)
                policy_kwargs = custom_obs_policy.create_augmented_nature_cnn(1)
                policy = sacCnn
            elif env0.depth_obs or env0.full_obs:
                policy_kwargs = {}
                policy = sacCnn
            else:
                policy_kwargs = {"net_arch": self.config[self.algo]['layers'], "layer_norm": False}
                policy = sacMlp
            if self.load_dir:
                top_folder_idx = self.load_dir.rfind('/')
                top_folder_str = self.load_dir[0:top_folder_idx]
                if self.norm:
                    self.env = VecNormalize(self.env, training=True, norm_obs=False, norm_reward=False, clip_obs=10.)
                    self.env = VecNormalize.load(os.path.join(top_folder_str, 'vecnormalize.pkl'), self.env)
                model = sb3.SAC(
                    policy,
                    self.env,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    gamma=self.config['discount_factor'],
                    buffer_size=self.config[self.algo]['buffer_size'],
                    batch_size=self.config[self.algo]['batch_size'],
                    learning_rate=self.config[self.algo]['step_size'],
                    tensorboard_log=tensorboard_file
                )
                model_load = sb3.SAC.load(self.load_dir, env=self.env)
                params = model_load.get_parameters()
                model.set_parameters(params)
            else:
                if self.norm:
                    self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True, clip_obs=10.)
                model = sb3.SAC(
                    policy,
                    self.env,
                    policy_kwargs=policy_kwargs,
                    verbose=2,
                    gamma=self.config['discount_factor'],
                    buffer_size=self.config[self.algo]['buffer_size'],
                    batch_size=self.config[self.algo]['batch_size'],
                    learning_rate=self.config[self.algo]['step_size'],
                    tensorboard_log=tensorboard_file
                )
        elif self.algo.upper() == 'PPO':
            from stable_baselines3 import PPO
            # Unwrap the environment.
            env0 = self.env.envs[0].unwrapped
            if not env0.is_simplified() and (env0.depth_obs or env0.full_obs):
                policy_kwargs = custom_obs_policy.create_augmented_nature_cnn(1)
                policy = sb3.ppo.policies.CnnPolicy
            elif env0.depth_obs or env0.full_obs:
                policy_kwargs = {}
                policy = sb3.ppo.policies.CnnPolicy
            else:
                policy_kwargs = {"net_arch": self.config[self.algo]['layers'], "layer_norm": False}
                policy = sb3.ppo.policies.MlpPolicy
            model = PPO(
                policy,
                self.env,
                verbose=2,
                gamma=self.config['discount_factor'],
                learning_rate=self.config[self.algo]['learning_rate'],
                tensorboard_log=tensorboard_file,
                policy_kwargs=policy_kwargs
            )
        elif self.algo.upper() == 'DQN':
            from stable_baselines3 import DQN
            from stable_baselines3.dqn.policies import MlpPolicy as DQNMlpPolicy
            if self.load_dir:
                model = self.load_params()
            else:
                model = DQN(
                    DQNMlpPolicy,
                    self.env,
                    verbose=2,
                    gamma=self.config['discount_factor'],
                    batch_size=self.config[self.algo]['batch_size'],
                    tensorboard_log=tensorboard_file
                )
        elif self.algo.upper() == 'TD3':
            from stable_baselines3 import TD3
            from stable_baselines3.td3.policies import MlpPolicy as TD3MlpPolicy
            model = TD3(
                TD3MlpPolicy,
                self.env,
                verbose=2,
                gamma=self.config['discount_factor'],
                learning_rate=self.config[self.algo]['step_size'],
                tensorboard_log=tensorboard_file
            )
        elif self.algo.upper() == 'A2C':
            from stable_baselines3 import A2C
            from stable_baselines3.a2c.policies import MlpPolicy as A2CMlpPolicy
            model = A2C(
                A2CMlpPolicy,
                self.env,
                verbose=2,
                gamma=self.config['discount_factor'],
                learning_rate=self.config[self.algo]['step_size'],
                tensorboard_log=tensorboard_file
            )
        else:
            raise Exception("Algorithm not supported in Stable-Baselines3.")
        
        try:
            model.learn(
                total_timesteps=int(self.config[self.algo]['total_timesteps']), 
                callback=[TensorboardCallback(self.env, self.log_freq, self.model_dir), 
                          eval_callback, checkpoint_callback, time_callback]
            )
        except KeyboardInterrupt:
            pass

        self.save(model, self.model_dir)

    def load_params(self):
        usable_params = {}
        print("Loading the model")
        from stable_baselines3 import DQN
        model_load = DQN.load(self.load_dir)
        pars = model_load.get_parameters()
        for key, value in pars.items():
            if 'action_value' not in key and '2' in key:
                usable_params.update({key: value})
        tensorboard_file = None
        model = DQN(
            sb3.dqn.policies.MlpPolicy,
            self.env,
            verbose=2,
            gamma=self.config['discount_factor'],
            batch_size=self.config[self.algo]['batch_size'],
            tensorboard_log=tensorboard_file
        )
        model.set_parameters(usable_params)
        return model

    def save(self, model, model_dir):
        if '/' in model_dir:
            top_folder, model_name = model_dir.split('/')
        else:
            model_name = model_dir
        folder_path = os.path.join(model_dir, model_name)
        if os.path.isfile(folder_path + '.zip'):
            print('File already exists')
            i = 1
            while os.path.isfile(f"{folder_path}_{i}.zip"):
                i += 1
            folder_path = f"{folder_path}_{i}"
            model.save(folder_path)
        else:
            print('Saving model to {}'.format(folder_path))
            model.save(folder_path)
        if self.norm:
            model.get_vec_normalize_env().save(os.path.join(model_dir, 'vecnormalize.pkl'))

