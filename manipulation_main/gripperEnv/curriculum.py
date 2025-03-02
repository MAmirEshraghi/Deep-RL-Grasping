"""
This script implements the WorkspaceCurriculum class, which adaptively adjusts the workspace parameters
(e.g., extent, robot height, number of objects, and lift distance) based on the agent's performance.
It is used during training to gradually increase the task difficulty and diversity of training samples.
"""
import collections  # For using deque to maintain a fixed-size history of episode outcomes.
import functools
import os

import numpy as np  # For numerical operations.
import pybullet  # For simulation-related operations (if needed).

from manipulation_main.gripperEnv import robot  # Import robot module to access RobotEnv.Status.

class WorkspaceCurriculum(object):
    """Adaptively adjusts workspace parameters to increase the diversity of training samples."""
    
    def __init__(self, config, robot, evaluate):
        """Initializes the curriculum using configuration parameters, the robot instance, and the evaluation flag."""
        self._robot = robot
        self._scene = robot._scene  # Reference to the current simulation scene.
        self._reward_fn = robot._reward_fn  # Reference to the reward function used by the robot.
        
        # Load curriculum configuration parameters.
        self._n_steps = config['n_steps']
        self._success_threshold = config['success_threshold']
        self._window_size = config['window_size']
        
        self._extent_range = config['extent']
        self._robot_height_range = config['robot_height']
        self._max_objects_range = config['max_objects']
        self._min_objects_range = config.get('min_objects', [1,1])
        self._work_range = config.get('workspace', None)
        self._work_height = config.get('work_height', None)
        
        self._lift_dist_range = config.get('lift_dist', None)
        
        # Initialize a history buffer to store recent episode successes.
        self._history = collections.deque(maxlen=self._window_size)
        # Set the initial lambda value for curriculum progression (1 if evaluating, else from config).
        self._lambda = config.get('init_lambda', 0.) if not evaluate else 1.
        self._update_parameters()  # Update parameters based on the initial lambda.
        
        self._policy_iteration = 1  # Counter for curriculum iterations.
    
    def update(self, task):
        """Updates the history with the current episode result and adjusts parameters if success rate is high."""
        # Append the success status of the current episode.
        self._history.append(task.status == robot.RobotEnv.Status.SUCCESS)
        
        # Wait until history is full before updating.
        if len(self._history) < self._history.maxlen:
            return
        # If the mean success rate exceeds the threshold and lambda is not yet maximum (1.0), update curriculum.
        if np.mean(self._history) > self._success_threshold and self._lambda is not 1.:
            # Increase lambda gradually.
            self._lambda = min(1., self._lambda + 1. / self._n_steps)
            self._update_parameters()  # Update workspace parameters based on the new lambda.
            self._history.clear()  # Clear the history after updating.
            print('Increased the step of the curriculum sequence to', self._lambda)
    
    def log_step(self, model_dir):
        """Logs the current policy iteration and lambda value to a CSV file for later analysis."""
        with open(os.path.join(model_dir, 'curriculum_steps.csv'), 'ba') as f:
            np.savetxt(f, [[self._policy_iteration, self._lambda]])
        self._policy_iteration += 1  # Increment the policy iteration counter.
    
    def _update_parameters(self):
        """Updates the workspace, scene, and reward parameters based on the current lambda value."""
        extent = _convert(self._lambda, self._extent_range)
        height = _convert(self._lambda, self._robot_height_range)
        max_objects = int(round(_convert(self._lambda, self._max_objects_range)))
        min_objects = int(round(_convert(self._lambda, self._max_objects_range)))  # Note: Both max and min use the same conversion.
        if self._work_range is not None:
            # Update the robot's workspace if workspace range is provided.
            workspace = _convert(self._lambda, self._work_range)
            work_height = _convert(self._lambda, self._work_height)
            self._robot._workspace = {'lower': np.array([-workspace, -workspace, -0.2]),
                                      'upper': np.array([workspace, workspace, work_height])}
            print("robot workspace", self._robot._workspace)
        
        # Update the scene parameters.
        self._scene.extent = extent
        self._scene.max_objects = max_objects
        self._scene.min_objects = min_objects
        self._robot._initial_height = height
        
        # Update the reward function's desired lift distance if specified.
        if self._lift_dist_range is not None:
            lift_dist = _convert(self._lambda, self._lift_dist_range)
            self._reward_fn.lift_dist = lift_dist

def _convert(val, new_range):
    """Converts a value in [0, 1] to a value in the new_range specified by [new_min, new_max]."""
    new_min, new_max = new_range[0], new_range[1]
    return new_min + (new_max - new_min) * val
