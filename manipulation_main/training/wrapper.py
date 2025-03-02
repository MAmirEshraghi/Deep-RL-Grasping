import gym
import numpy as np
from gym.wrappers import TimeLimit

class TimeFeatureWrapper(gym.Wrapper):
    """
    Gym wrapper that augments the observation with a time feature indicating
    the remaining fraction of time in an episode. This is useful in fixed-length
    episodes so that the agent is aware of how much time is left.

    References:
      - https://arxiv.org/abs/1712.00378
      - https://github.com/aravindr93/mjrl/issues/13

    Args:
        env (gym.Env): The environment to wrap.
        max_steps (int): Maximum number of steps per episode if the environment is
                         not already wrapped by a TimeLimit.
        test_mode (bool): When True, the time feature remains constant at 1.0 to
                          prevent the agent from overfitting to the time feature.
    """
    def __init__(self, env, max_steps=1000, test_mode=False):
        # Ensure that the environment's observation space is a Box space,
        # which supports continuous numerical data.
        assert isinstance(env.observation_space, gym.spaces.Box)
        
        # Retrieve the original lower and upper bounds of the observation space.
        low, high = env.observation_space.low, env.observation_space.high
        
        # Extend these bounds by one dimension to include the time feature.
        # The time feature is normalized between 0 and 1.
        low, high = np.concatenate((low, [0])), np.concatenate((high, [1.]))
        
        # Update the observation space to include the new time feature dimension.
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Initialize the base gym wrapper with the modified environment.
        super(TimeFeatureWrapper, self).__init__(env)
        
        # If the environment is already limited by a TimeLimit wrapper,
        # use its maximum number of steps; otherwise, use the provided max_steps.
        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        
        # Initialize the step counter for the current episode.
        self._current_step = 0
        
        # Store the test_mode flag.
        self._test_mode = test_mode

    def reset(self):
        """
        Reset the environment and the internal step counter.
        Return the initial observation augmented with the time feature.
        """
        # Reset the step counter for the new episode.
        self._current_step = 0
        
        # Reset the underlying environment and augment its initial observation.
        return self._get_obs(self.env.reset())

    def step(self, action):
        """
        Execute an action in the environment, increment the step counter,
        and return the new observation (with time feature), reward, done flag,
        and any additional information.
        """
        # Increment the current step count.
        self._current_step += 1
        
        # Perform the action in the original environment.
        obs, reward, done, info = self.env.step(action)
        
        # Return the observation augmented with the time feature along with other outputs.
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Augment the given observation with the time feature.

        Args:
            obs (np.ndarray): Original observation from the environment.
        
        Returns:
            np.ndarray: Augmented observation including the time feature.
        """
        # Compute the time feature as the fraction of time remaining.
        # It is calculated as: 1 - (current_step / max_steps).
        time_feature = 1 - (self._current_step / self._max_steps)
        
        # If in test mode, override the time feature to remain constant at 1.0.
        if self._test_mode:
            time_feature = 1.0
        
        # Concatenate the time feature to the original observation.
        return np.concatenate((obs, [time_feature]))
