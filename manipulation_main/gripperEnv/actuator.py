"""
This script implements the Actuator class, which converts high-level action commands into low-level robot movements 
(e.g., translations, rotations, and gripper actions) and manages action/observation space setup and scaling.
It is used by the simulation environment (RobotEnv) to execute the actions determined by the RL agent.
"""

import numpy as np  # Numerical computations.
import gymnasium as gym # For defining Gym-compatible action and observation spaces.

from sklearn.preprocessing import MinMaxScaler  # To scale actions into a normalized range.

class Actuator:
    """Handles the conversion of normalized actions into robot movements and gripper operations."""
    
    def __init__(self, robot, config, simplified):
        """Initializes the actuator with robot reference, configuration parameters, and mode (simplified/full)."""
        self.robot = robot
        self._include_robot_height = config.get('include_robot_height', False)
        self._simplified = simplified

        # Define maximum translation, yaw rotation, and force based on configuration.
        self._max_translation = config['robot']['max_translation']
        self._max_yaw_rotation = config['robot']['max_yaw_rotation']
        self._max_force = config['robot']['max_force']

        # Determine if the action space is discrete and set discrete step sizes if so.
        self._discrete = config['robot']['discrete']
        self._discrete_step = config['robot']['step_size']
        self._yaw_step = config['robot']['yaw_step']
        if self._discrete:
            self.num_actions_pad = config['robot']['num_actions_pad']
            self.num_act_grains = self.num_actions_pad - 1
            self.trans_action_range = 2 * self._max_translation
            self.yaw_action_range = 2 * self._max_yaw_rotation

        # Initialize the gripper state to open.
        self._gripper_open = True
        self.state_space = None

    def reset(self):
        """Resets the actuator state by opening the robot gripper and setting its state to open."""
        self.robot.open_gripper()
        self._gripper_open = True

    def step(self, action):
        """Processes the incoming action by denormalizing it (if needed) and executing it."""
        if not self._discrete:
            action = self._action_scaler.inverse_transform(np.array([action]))
            action = action.squeeze()
        # Execute the action using the appropriate low-level function.
        return self._act(action)

    def get_state(self):
        """Returns the current state of the gripper (and optionally robot height) scaled to [0, 1]."""
        if self._include_robot_height:
            gripper_width = self.robot.get_gripper_width()
            position, _ = self.robot.get_pose()
            height = position[2]
            state = self._obs_scaler * np.r_[gripper_width, height]
        else:
            state = self._obs_scaler * self.robot.get_gripper_width()
        return state

    def setup_action_space(self):
        """Sets up and returns the Gym-compatible action space based on the mode (simplified or full) and discrete flag."""
        if self._simplified:
            high = np.r_[[self._max_translation] * 2, self._max_yaw_rotation]
            self._action_scaler = MinMaxScaler((-1, 1))
            self._action_scaler.fit(np.vstack((-1. * high, high)))
            if not self._discrete:
                self.action_space = gym.spaces.Box(-1., 1., shape=(3,), dtype=np.float32)
            else:
                self.action_space = gym.spaces.Discrete(self.num_actions_pad*3)
            self._act = self._simplified_act
        else:
            high = np.r_[[self._max_translation] * 3, self._max_yaw_rotation, 1.]
            self._action_scaler = MinMaxScaler((-1, 1))
            self._action_scaler.fit(np.vstack((-1. * high, high)))
            if not self._discrete:
                self.action_space = gym.spaces.Box(-1., 1., shape=(5,), dtype=np.float32)
            else:
                self.action_space = gym.spaces.Discrete(11)
                # TODO: Implement the linear discretization for full environment.
            self._act = self._full_act

            if self._include_robot_height:
                self._obs_scaler = np.array([1. / 0.05, 1.])
                self.state_space = gym.spaces.Box(0., 1., shape=(2,), dtype=np.float32)
            else:
                self._obs_scaler = 1. / 0.1
                self.state_space = gym.spaces.Box(0., 1., shape=(1,), dtype=np.float32)
        return self.action_space

    def _clip_translation_vector(self, translation, yaw):
        """Clips the translation vector and yaw if they exceed maximum allowed values."""
        length = np.linalg.norm(translation)
        if length > self._max_translation:
            translation *= self._max_translation / length
        if yaw > self._max_yaw_rotation:
            yaw *= self._max_yaw_rotation / yaw
        return translation, yaw

    def _full_act(self, action):
        """Executes the full action by parsing the action vector (or discrete index) and applying translation, rotation, and gripper commands."""
        if not self._discrete:
            translation, yaw_rotation = self._clip_translation_vector(action[:3], action[3])
            open_close = action[4]
        else:
            assert(isinstance(action, (np.int64, int)))
            x = [0, self._discrete_step, -self._discrete_step, 0, 0, 0, 0, 0, 0, 0, 0][action]
            y = [0, 0, 0, self._discrete_step, -self._discrete_step, 0, 0, 0, 0, 0, 0][action]
            z = [0, 0, 0, 0, 0, self._discrete_step, -self._discrete_step, 0, 0, 0, 0][action]
            a = [0, 0, 0, 0, 0, 0, 0, self._yaw_step, -self._yaw_step, 0, 0 ][action]
            open_close = [0, 0, 0, 0, 0, 0, 0, 0, 0, self._discrete_step, -self._discrete_step][action]
            translation = [x, y, z]
            yaw_rotation = a
        if open_close > 0. and not self._gripper_open:
            self.robot.open_gripper()
            self._gripper_open = True
        elif open_close < 0. and self._gripper_open:
            self.robot.close_gripper()
            self._gripper_open = False
        else:
            return self.robot.relative_pose(translation, yaw_rotation)

    def _simplified_act(self, action):
        """Executes the simplified action by parsing a shorter action vector (or discrete index) and applying a 2D translation with yaw."""
        if not self._discrete:
            translation, yaw_rotation = self._clip_translation_vector(action[:2], action[2])
        else:
            assert(isinstance(action, (np.int64, int)))
            if action < self.num_actions_pad:
                x = action / self.num_act_grains * self.trans_action_range - self._max_translation
                y = 0
                a = 0
            elif self.num_actions_pad <= action < 2*self.num_actions_pad:
                action -= self.num_actions_pad
                x = 0
                y = action / self.num_act_grains * self.trans_action_range - self._max_translation
                a = 0
            else:
                action -= 2*self.num_actions_pad
                x = 0
                y = 0
                a = action / self.num_act_grains * self.yaw_action_range - self._max_yaw_rotation
            translation = [x, y]
            yaw_rotation = a
        translation = np.r_[translation, 0.005]  # Add a constant Z offset.
        return self.robot.relative_pose(translation, yaw_rotation)

    def is_discrete(self):
        """Returns True if the actuator uses a discrete action space, otherwise False."""
        return self._discrete
