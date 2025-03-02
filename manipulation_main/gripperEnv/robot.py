"""
The RobotEnv class defines a robotics simulation environment for a gripper in a PyBullet world.
It is part of a modular framework that separates different responsibilities into various scripts:

- Common Utilities (io_utils, transformations, transform_utils):
These modules provide functions to load configurations, handle transformations (e.g., converting between Euler angles and quaternions), and other helper routines.

- Sensors and Actuators (sensor, actuator):
The sensor module provides the code to handle RGB-D or encoded depth sensors. The actuator module provides the code to control the robot’s motors (including gripper control).

- Simulation (World):
The World class (imported from manipulation_main/simulation/simulation.py) provides a base simulation environment that RobotEnv extends.

- Rewards (Reward, SimplifiedReward, ShapedCustomReward):
These modules define different reward functions that shape the learning signal for the RL agent.

- Curriculum (WorkspaceCurriculum):
This module manages curriculum learning by updating workspace parameters based on the agent’s performance.

- Reset Helper (_reset):
The helper function _reset repeatedly resets the simulation until the sensor (depth sensor) detects an object (or until empty states are not skipped).
"""


import os
import time
import pybullet as p
import numpy as np
import functools
import gym
import collections
from gym import spaces 
from enum import Enum

# Import common utilities for IO, transformations, etc.
from manipulation_main.common import io_utils
from manipulation_main.common import transformations
from manipulation_main.common import transform_utils
# Import sensor and actuator modules for the gripper environment.
from manipulation_main.gripperEnv import sensor, actuator #(for perception and action)
# Import the base World simulation class.
from manipulation_main.simulation.simulation import World #(simulation backbone)
# Import different reward functions.
from manipulation_main.gripperEnv.rewards import Reward, SimplifiedReward, ShapedCustomReward
# Import the workspace curriculum for adaptive learning.
from manipulation_main.gripperEnv.curriculum import WorkspaceCurriculum


def _reset(robot, actuator, depth_sensor, skip_empty_states=False):
    """Reset until an object is within the fov of the camera."""
    """
    Helper function to reset the simulation until an object is within the
    field of view of the depth sensor.
    
    Args:
        robot: The robot environment instance (RobotEnv).
        actuator: The actuator controlling the robot.
        depth_sensor: The depth sensor (camera) of the robot.
        skip_empty_states (bool): If False, accept any state regardless of sensor reading.
    """
    ok = False
    while not ok:
        # Reset the simulation world and scene.
        robot.reset_sim()  # Resets the world and scene
        robot.reset_model()  # Resets the robot model
        actuator.reset()  # Resets the actuator state
        
        # Get the sensor state; we expect a mask from the depth sensor.
        _, _, mask = depth_sensor.get_state()
        
        # Check that more than two unique values exist in the mask (i.e., object is detected).
        ok = len(np.unique(mask)) > 2  # Plane and gripper are always visible
        
        # If not skipping empty states, then accept the current state.
        if not skip_empty_states:
            ok = True

class RobotEnv(World):

    class Events(Enum):
        START_OF_EPISODE = 0
        END_OF_EPISODE = 1
        CLOSE = 2
        CHECKPOINT = 3

    class Status(Enum):
        RUNNING = 0
        SUCCESS = 1
        FAIL = 2
        TIME_LIMIT = 3

    def __init__(self, config, evaluate=False, test=False, validate=False):
        """
        Initialize the Robot Environment.
        
        Args:
            config (dict or str): Configuration dictionary or path to YAML file.
            evaluate (bool): Whether the environment is in evaluation mode.
            test (bool): Whether in testing mode.
            validate (bool): Whether in validation mode.
        """
        # Load configuration from YAML if needed.
        if not isinstance(config, dict):
            config = io_utils.load_yaml(config)
        
        # Initialize the base World with configuration and mode flags.
        super().__init__(config, evaluate=evaluate, test=test, validate=validate)
        
        # Store timing information with a large history buffer.
        self._step_time = collections.deque(maxlen=10000)
        
        # Define the episode time horizon from the configuration.
        self.time_horizon = config['time_horizon']
        
        # Define workspace boundaries (a 3D cube).
        self._workspace = {'lower': np.array([-1., -1., -1]),
                           'upper': np.array([1., 1., 1.])}
        
        # Path to the robot model.
        self.model_path = config['robot']['model_path']
        
        # Simplified mode flag from configuration.
        self._simplified = config['simplified']
        
        # Flags for observation modes.
        self.depth_obs = config.get('depth_observation', False)
        self.full_obs = config.get('full_observation', False)
        
        # Initial height for the robot model.
        self._initial_height = 0.3
        
        # Initial orientation as a quaternion (rotated π radians about the X-axis).
        self._init_ori = transformations.quaternion_from_euler(np.pi, 0., 0.)
        
        # Define main joints (hardcoded indices; improvement needed in future).
        self.main_joints = [0, 1, 2, 3]
        
        # Define finger joint IDs.
        self._left_finger_id = 7
        self._right_finger_id = 9
        self._fingers = [self._left_finger_id, self._right_finger_id]

        # Initialize placeholders for model and joint objects.
        self._model = None
        self._joints = None
        self._left_finger, self._right_finger = None, None
        
        # Initialize the actuator with the environment, configuration, and simplified flag.
        self._actuator = actuator.Actuator(self, config, self._simplified)

        # Initialize the RGB-D sensor (camera) with sensor configuration.
        self._camera = sensor.RGBDSensor(config['sensor'], self)

        # Select and assign the reward function based on configuration.
        if self._simplified:
            self._reward_fn = SimplifiedReward(config['reward'], self)
        elif config['reward']['custom']:
            self._reward_fn = ShapedCustomReward(config['reward'], self)
        else:    
            self._reward_fn = Reward(config['reward'], self)

        # Setup the sensors for observation.
        if self.depth_obs or self.full_obs:
            # If using raw depth or full observation (e.g., RGB-D), use the camera.
            self._sensors = [self._camera]
        else:
            # Otherwise, use an encoded depth sensor.
            self._encoder = sensor.EncodedDepthImgSensor(config, self._camera, self)
            self._sensors = [self._encoder]
        
        # If not in simplified mode, include actuator state as an observation.
        if not self._simplified:
            self._sensors.append(self._actuator)

        # Initialize the workspace curriculum for adaptive learning.
        self.curriculum = WorkspaceCurriculum(config['curriculum'], self, evaluate)

        # Store the curriculum history.
        self.history = self.curriculum._history
        
        # Initialize a dictionary for event callbacks.
        self._callbacks = {RobotEnv.Events.START_OF_EPISODE: [],
                           RobotEnv.Events.END_OF_EPISODE: [],
                           RobotEnv.Events.CLOSE: [],
                           RobotEnv.Events.CHECKPOINT: []}
        
        # Register default event callbacks.
        self.register_events(evaluate, config)
        
        # Initialize the mean success rate.
        self.sr_mean = 0.
        
        # Setup the Gym action and observation spaces.
        self.setup_spaces()

    def register_events(self, evaluate, config):
        """
        Register callbacks for various events in the episode.
        
        Args:
            evaluate (bool): Evaluation mode flag.
            config (dict): Configuration dictionary.
        """
        # Determine whether to skip empty states based on evaluation mode.
        skip_empty_states = True if evaluate else config['skip_empty_initial_state']
        # Create a partial function for reset that uses _reset with fixed parameters.
        reset = functools.partial(_reset, self, self._actuator, self._camera, skip_empty_states)

        # Register the reset callback at the start of the episode.
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, reset)
        # Also register the camera and reward function reset callbacks.
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self._camera.reset)
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self._reward_fn.reset)
        # Register curriculum update at the end of the episode.
        self.register_callback(RobotEnv.Events.END_OF_EPISODE, self.curriculum.update)
        # Register the close callback using the parent class's close method.
        self.register_callback(RobotEnv.Events.CLOSE, super().close)

    def reset(self):
        """
        Reset the environment for a new episode.
        
        Triggers start-of-episode events, resets counters, and obtains the initial observation.
        
        Returns:
            The initial observation.
        """
        # Trigger all callbacks associated with the start of an episode.
        self._trigger_event(RobotEnv.Events.START_OF_EPISODE)
        self.episode_step = 0
        # Initialize an array to store rewards for the episode.
        self.episode_rewards = np.zeros(self.time_horizon)
        # Set the status to running.
        self.status = RobotEnv.Status.RUNNING
        # Get the current observation.
        self.obs = self._observe()
        return self.obs

    def reset_model(self):
        """
        Reset the task.
        Reset the robot model within the simulation.
    
        Loads the robot model at the initial position and orientation, and sets up joint references.
        """
        """
        Returns:
            Observation of the initial state.
        """
        self.endEffectorAngle = 0.
        start_pos = [0., 0., self._initial_height]
        # Add the robot model to the simulation.
        self._model = self.add_model(self.model_path, start_pos, self._init_ori)
        self._joints = self._model.joints
        self.robot_id = self._model.model_id
        # Set references to the left and right finger joints.
        self._left_finger = self._model.joints[self._left_finger_id]
        self._right_finger = self._model.joints[self._right_finger_id]

    def _trigger_event(self, event, *event_args):
        """
        Trigger all registered callbacks for a given event.
        
        Args:
            event: An event from RobotEnv.Events.
            event_args: Additional arguments passed to the callback functions.
        """
        for fn, args, kwargs in self._callbacks[event]:
            fn(*(event_args + args), **kwargs)

    def register_callback(self, event, fn, *args, **kwargs):
        """
        Register a callback function for a specific event.
        
        Args:
            event: The event to register the callback for.
            fn: The callback function.
            *args, **kwargs: Additional arguments for the callback.
        """
        self._callbacks[event].append((fn, args, kwargs))

    def step(self, action):
        """
        Advance the simulation by one step using the given action.
        
        Args:
            action (np.ndarray): The action to execute.
            
        Returns:
            A tuple (obs, reward, done, info), where:
              - obs: The new observation.
              - reward: The reward obtained.
              - done: A boolean indicating whether the episode has ended.
              - info: A dictionary with additional information.
        """

        # If the robot model hasn't been initialized, reset the environment.
        if self._model is None:
            self.reset()

        # Execute the action using the actuator.
        self._actuator.step(action)

        # Get the new observation after the action.
        new_obs = self._observe()

        # Calculate the reward and update the status based on the previous and new observations.
        reward, self.status = self._reward_fn(self.obs, action, new_obs)
        self.episode_rewards[self.episode_step] = reward

        # Check termination conditions: if the status is no longer running or time horizon is reached.
        if self.status != RobotEnv.Status.RUNNING:
            done = True
        elif self.episode_step == self.time_horizon - 1:
            done, self.status = True, RobotEnv.Status.TIME_LIMIT
        else:
            done = False

        # Trigger the end-of-episode event if the episode is done.
        if done:
            self._trigger_event(RobotEnv.Events.END_OF_EPISODE, self)

        # Increment the episode step counter.
        self.episode_step += 1
        self.obs = new_obs
        # Update mean success rate based on curriculum history.
        if len(self.curriculum._history) != 0:
            self.sr_mean = np.mean(self.curriculum._history)
        # Advance the simulation in the parent World class.
        super().step_sim()
        
        # Return the observation, reward, done flag, and additional info.
        return self.obs, reward, done, {"is_success": self.status == RobotEnv.Status.SUCCESS,
                                        "episode_step": self.episode_step,
                                        "episode_rewards": self.episode_rewards,
                                        "status": self.status}

    def _observe(self):
        """
        Collect the current observation from all sensors.
        
        Returns:
            The combined observation.
        """
        # If not using depth or full observation, simply concatenate sensor states.
        if not self.depth_obs and not self.full_obs:
            obs = np.array([])
            for sensor in self._sensors:
                obs = np.append(obs, sensor.get_state())
            return obs
        else:
            # Retrieve RGB, depth, and any additional sensor info from the camera.
            rgb, depth, _ = self._camera.get_state()
            sensor_pad = np.zeros(self._camera.state_space.shape[:2])
            if self._simplified:
                # In simplified mode, stack depth with a sensor pad.
                obs_stacked = np.dstack((depth, sensor_pad))
                return obs_stacked

            # In non-simplified mode, use actuator state as a sensor pad.
            sensor_pad = np.zeros(self._camera.state_space.shape[:2])
            sensor_pad[0][0] = self._actuator.get_state()
            if self.full_obs:
                # If full observation is enabled, stack RGB, depth, and actuator state.
                obs_stacked = np.dstack((rgb, depth, sensor_pad))
            else:
                # Otherwise, stack only depth and actuator state.
                obs_stacked = np.dstack((depth, sensor_pad))
            return obs_stacked

    def setup_spaces(self):
        """
        Set up the Gym action and observation spaces based on sensor and actuator configurations.
        """
        # Set up the action space using the actuator.
        self.action_space = self._actuator.setup_action_space()
        if not self.depth_obs and not self.full_obs:
            # Concatenate state spaces from all sensors.
            low, high = np.array([]), np.array([])
            for sensor in self._sensors:
                low = np.append(low, sensor.state_space.low)
                high = np.append(high, sensor.state_space.high)
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        else:
            # For visual observations, set up the observation space based on the camera's state space.
            shape = self._camera.state_space.shape
            if self._simplified:
                # Simplified mode: observation space has 2 channels (e.g., depth and pad).
                self.observation_space = gym.spaces.Box(low=0, high=255,
                                                        shape=(shape[0], shape[1], 2))
            else:
                if self.full_obs:  # Full observation: RGB + Depth + Actuator state.
                    self.observation_space = gym.spaces.Box(low=0, high=255,
                                                            shape=(shape[0], shape[1], 5))
                else:  # Otherwise, Depth + Actuator observation.
                    self.observation_space = gym.spaces.Box(low=0, high=255,
                                                            shape=(shape[0], shape[1], 2))

    def reset_robot_pose(self, target_pos, target_orn):
        """
        Reset the robot base's pose in the world. Useful for testing.
        
        Args:
            target_pos (list): The target position [x, y, z].
            target_orn: The target orientation.
        """
        self.reset_base(self._model.model_id, target_pos, target_orn)
        self.run(0.1)


    def absolute_pose(self, target_pos, target_orn):
        """
        Set the robot's pose absolutely based on a target position and orientation.
        
        Args:
            target_pos (list): The target position.
            target_orn: The target orientation (or yaw angle).
        """
        # Adjust target position for coordinate system differences.
        target_pos[1] *= -1
        target_pos[2] = -1 * (target_pos[2] - self._initial_height)
        
        # Use target_orn as yaw; combine position and yaw.
        yaw = target_orn
        comp_pos = np.r_[target_pos, yaw]

        # Update the positions of the main joints.
        for i, joint in enumerate(self.main_joints):
            self._joints[joint].set_position(comp_pos[i])
        
        self.run(0.1)

    def relative_pose(self, translation, yaw_rotation):
        """
        Update the robot's pose relative to its current pose using a translation and yaw rotation.
        
        Args:
            translation (list): Translation vector.
            yaw_rotation (float): Rotation around the yaw axis.
        """
        pos, orn = self._model.get_pose()
        _, _, yaw = transform_utils.euler_from_quaternion(orn)
        # Compose the current transformation matrix.
        T_world_old = transformations.compose_matrix(angles=[np.pi, 0., yaw], translate=pos)
        # Compose the relative transformation matrix.
        T_old_to_new = transformations.compose_matrix(angles=[0., 0., yaw_rotation], translate=translation)
        # Compute the new transformation matrix.
        T_world_new = np.dot(T_world_old, T_old_to_new)
        self.endEffectorAngle += yaw_rotation
        target_pos, target_orn = transform_utils.to_pose(T_world_new)
        self.absolute_pose(target_pos, self.endEffectorAngle)


    def close_gripper(self):
        """
        Close the gripper by setting the target joint positions for the fingers.
        """
        self.gripper_close = True
        self._target_joint_pos = 0.05
        self._left_finger.set_position(self._target_joint_pos)
        self._right_finger.set_position(self._target_joint_pos)
        self.run(0.2)

    def open_gripper(self):
        """
        Open the gripper by setting the target joint positions for the fingers.
        """
        self.gripper_close = False
        self._target_joint_pos = 0.0
        self._left_finger.set_position(self._target_joint_pos)
        self._right_finger.set_position(self._target_joint_pos)
        self.run(0.2)

    def _enforce_constraints(self, position):
        """
        Enforce workspace constraints on a given position.
        
        Args:
            position (np.ndarray): The desired position.
            
        Returns:
            The position clipped to the workspace bounds.
        """
        if self._workspace:
            position = np.clip(position,
                               self._workspace['lower'],
                               self._workspace['upper'])
        return position
    
    def get_gripper_width(self):
        """
        Query the current opening width of the gripper.
        
        Returns:
            The combined width of the left and right fingers.
        """
        left_finger_pos = 0.05 - self._left_finger.get_position()
        right_finger_pos = 0.05 - self._right_finger.get_position()
        return left_finger_pos + right_finger_pos

    def object_detected(self, tol=0.005):
        """
        Check if an object is detected by examining if the gripper fingers have stalled while closing.
        
        Args:
            tol (float): Tolerance for gripper width.
            
        Returns:
            True if an object is detected; False otherwise.
        """
        return self._target_joint_pos == 0.05 and self.get_gripper_width() > tol

    def get_pose(self):
        """
        Get the current pose of the robot model.
        
        Returns:
            The pose as provided by the robot model.
        """
        return self._model.get_pose()
    def is_simplified(self):
        """
        Check if the environment is in simplified mode.
        
        Returns:
            True if simplified; False otherwise.
        """
        return self._simplified

    def is_discrete(self):
        """
        Check if the actuator uses a discrete action space.
        
        Returns:
            True if discrete; False otherwise.
        """
        return self._actuator.is_discrete()