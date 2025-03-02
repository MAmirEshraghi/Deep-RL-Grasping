"""
This script implements reward functions for the robotic grasping environment.
It provides three reward strategies:
  - Reward: a basic function that reinforces upward movement when an object is grasped,
  - SimplifiedReward: a version for simplified environments focusing on a direct grasp attempt,
  - ShapedCustomReward: a customized reward that builds on Reward with additional table-clearing logic.
These functions are invoked during simulation steps to compute the reward signal and determine episode status.
"""




from manipulation_main.gripperEnv import robot  # Import robot module to access RobotEnv.Status.

class Reward:
    """Simple reward function reinforcing upwards movement of grasped objects."""
    
    def __init__(self, config, robot):
        """Initializes the Reward function using configuration parameters and the robot instance."""
        self._robot = robot
        self._shaped = config.get('shaped', True)
        self._max_delta_z = robot._actuator._max_translation  # Maximum change in height allowed.
        self._terminal_reward = config['terminal_reward']  # Reward given at episode termination.
        self._grasp_reward = config['grasp_reward']  # Base reward for grasping.
        self._delta_z_scale = config['delta_z_scale']  # Scaling factor for height difference.
        self._lift_success = config.get('lift_success', self._terminal_reward)  # Reward for successful lift.
        self._time_penalty = config.get('time_penalty', False)  # Penalty applied per time step.
        self._table_clearing = config.get('table_clearing', False)  # Flag for additional table-clearing reward.
        self.lift_dist = None  # Desired lift distance (to be set externally).
        
        # Placeholders for tracking lift progress.
        self._lifting = False  
        self._start_height = None
        self._old_robot_height = None

    def __call__(self, obs, action, new_obs):
        """Computes the reward and episode status based on the robot's vertical movement and grasp detection."""
        position, _ = self._robot.get_pose()
        robot_height = position[2]
        reward = 0.
        
        # Check if the robot has detected an object.
        if self._robot.object_detected():
            if not self._lifting:
                self._start_height = robot_height  # Start measuring lift distance.
                self._lifting = True
            if robot_height - self._start_height > self.lift_dist:
                # Object was lifted by the desired amount: return terminal reward.
                return self._terminal_reward, robot.RobotEnv.Status.SUCCESS
            if self._shaped:
                # Provide intermediate reward based on change in robot height.
                delta_z = robot_height - self._old_robot_height
                reward = self._grasp_reward + self._delta_z_scale * delta_z
        else:
            self._lifting = False

        # Apply a time penalty for each step if using shaped rewards.
        if self._shaped:
            reward -= self._grasp_reward + self._delta_z_scale * self._max_delta_z
        else:
            reward -= 0.01

        self._old_robot_height = robot_height
        return reward, robot.RobotEnv.Status.RUNNING

    def reset(self):
        """Resets internal height tracking to the current robot height."""
        position, _ = self._robot.get_pose()
        self._old_robot_height = position[2]


class SimplifiedReward:
    """Reward function for the simplified RobotEnv focusing on a direct grasp attempt."""
    
    def __init__(self, config, robot):
        """Initializes the SimplifiedReward with the robot instance and configuration flags."""
        self._robot = robot
        self._old_robot_height = None
        self._stalled_act = config.get('stalled', True)

    def __call__(self, obs, action, new_obs):
        """Computes a simplified reward by directly checking target height and grasp success."""
        position, _ = self._robot.get_pose()
        robot_height = position[2]
        if robot_height < 0.07:
            # If robot height is below target, attempt to grasp the object.
            self._robot.close_gripper()
            if not self._robot.object_detected():
                return 0., robot.RobotEnv.Status.FAIL
            # Move downward slightly multiple times.
            for _ in range(10):
                self._robot.relative_pose([0., 0., -0.005], 0.)
            if self._robot.object_detected():
                return 1., robot.RobotEnv.Status.SUCCESS
            else:
                return 0., robot.RobotEnv.Status.FAIL

        elif (self._old_robot_height - robot_height < 0.002) and self._stalled_act:
            # If there is minimal downward movement and actions are stalled, mark as failure.
            return 0., robot.RobotEnv.Status.FAIL

        else:
            # Otherwise, update the robot height and continue running.
            self._old_robot_height = robot_height
            return 0., robot.RobotEnv.Status.RUNNING

    def reset(self):
        """Resets the internal height tracker to the current robot height."""
        position, _ = self._robot.get_pose()
        self._old_robot_height = position[2]


class ShapedCustomReward(Reward):
    """Customized reward function that extends Reward with table-clearing behavior and adjusted lifting rewards."""
    
    def __call__(self, obs, action, new_obs):
        """Computes a shaped reward that includes additional logic for table clearing and multiple-object grasping."""
        position, _ = self._robot.get_pose()
        robot_height = position[2]
        reward = 0.

        if self._robot.object_detected():
            if not self._lifting:
                self._start_height = robot_height  # Start measuring lift distance.
                self._lifting = True

            if robot_height - self._start_height > self.lift_dist:
                if self._table_clearing:
                    # If table clearing is enabled, remove the object from the scene.
                    grabbed_obj = self._robot.find_highest()
                    if grabbed_obj is not -1:
                        self._robot.remove_model(grabbed_obj)
                    
                    # Alternative: remove multiple objects if needed (code commented out).
                    self._robot.open_gripper()
                    if self._robot.get_num_body() == 2: 
                        return self._terminal_reward, robot.RobotEnv.Status.SUCCESS
                    return self._lift_success, robot.RobotEnv.Status.RUNNING
                else:
                    if not self._shaped:
                        return 1., robot.RobotEnv.Status.SUCCESS
                    return self._terminal_reward, robot.RobotEnv.Status.SUCCESS
            if self._shaped:
                # Provide intermediate reward based on vertical movement.
                delta_z = robot_height - self._old_robot_height
                reward = self._grasp_reward + self._delta_z_scale * delta_z
        else:
            self._lifting = False

        # Apply a time penalty if using shaped rewards.
        if self._shaped:
            reward -= self._time_penalty
        else:
            reward -= 0.01

        self._old_robot_height = robot_height
        return reward, robot.RobotEnv.Status.RUNNING
