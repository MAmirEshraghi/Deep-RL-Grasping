"""
This script implements the World class, a Gym-compliant simulation environment based on PyBullet.
It manages the physics simulation, loads models, sets up scenes (e.g., OnTable or OnFloor), and provides
utility methods for stepping, resetting, and querying the simulation. This module serves as the backbone
of the simulated world for the robotic grasping tasks.
"""
from enum import Enum  # For defining simulation event types.
import pybullet as p  # PyBullet physics engine.
import time  # For time-related functions.
import gym  # Gym interface for environments.

from gym.utils import seeding  # For creating reproducible random seeds.
from numpy.random import RandomState  # For random number generation.
from manipulation_main.simulation.model import Model  # Model loader for simulation objects.
from manipulation_main.simulation import scene  # Scene definitions (e.g., OnTable, OnFloor).
from pybullet_utils import bullet_client  # Wrapper for managing PyBullet connections.

class World(gym.Env):
    """Defines a Gym-compliant simulation world using PyBullet as the physics engine."""
    
    class Events(Enum):
        RESET = 0  # Event type for simulation reset.
        STEP = 1   # Event type for each simulation step.

    def __init__(self, config, evaluate, test, validate):
        """
        Initializes a new simulated world based on the configuration and mode flags.
        
        Args:
            config (dict): Configuration dictionary with simulation parameters.
            evaluate (bool): Flag indicating evaluation mode.
            test (bool): Flag indicating test mode.
            validate (bool): Flag indicating validation mode.
        """
        self._rng = self.seed(evaluate=evaluate)  # Initialize the RNG for reproducibility.
        config_scene = config['scene']
        self.scene_type = config_scene.get('scene_type', "OnTable")  # Determine the scene type.
        # Create the appropriate scene based on scene_type.
        if self.scene_type == "OnTable":
            self._scene = scene.OnTable(self, config, self._rng, test, validate)
        elif self.scene_type == "OnFloor":
            self._scene = scene.OnFloor(self, config, self._rng, test, validate)
        else:
            self._scene = scene.OnTable(self, config, self._rng, test, validate)
        
        self.sim_time = 0.  # Initialize simulation time.
        self._time_step = 1. / 240.  # Set the simulation time step.
        self._solver_iterations = 150  # Set the number of solver iterations.

        config = config['simulation']
        visualize = config.get('visualize', True)  # Determine if visualization is enabled.
        self._real_time = config.get('real_time', True)  # Determine if simulation should run in real time.
        # Create a PyBullet client with GUI or DIRECT mode based on visualization flag.
        self.physics_client = bullet_client.BulletClient(
            p.GUI if visualize else p.DIRECT)

        self.models = []  # List to store models loaded into the simulation.
        # Initialize callbacks for simulation events.
        self._callbacks = {World.Events.RESET: [], World.Events.STEP: []}

    def run(self, duration):
        """Runs the simulation for the specified duration by repeatedly stepping the simulation."""
        for _ in range(int(duration / self._time_step)):
            self.step_sim()

    def add_model(self, path, start_pos, start_orn, scaling=1.):
        """Loads a model from file, adds it to the simulation, and returns the model instance."""
        model = Model(self.physics_client)
        model.load_model(path, start_pos, start_orn, scaling)
        self.models.append(model)
        return model

    def step_sim(self):
        """Advances the simulation by one step and synchronizes with real time if enabled."""
        self.physics_client.stepSimulation()
        # Uncomment the following line to trigger step events:
        # self._trigger_event(World.Events.STEP)
        self.sim_time += self._time_step
        if self._real_time:
            # Sleep to synchronize simulation time with real time.
            time.sleep(max(0., self.sim_time - time.time() + self._real_start_time))

    def reset_sim(self):
        """Resets the simulation by reinitializing the physics engine, gravity, and scene, and resets simulation time."""
        # Uncomment the following line to trigger reset events:
        # self._trigger_event(World.Events.RESET)
        self.physics_client.resetSimulation()
        self.physics_client.setPhysicsEngineParameter(
            fixedTimeStep=self._time_step,
            numSolverIterations=self._solver_iterations,
            enableConeFriction=1)
        self.physics_client.setGravity(0., 0., -9.81)
        self.models = []
        self.sim_time = 0.
        self._real_start_time = time.time()  # Record the real start time.
        self._scene.reset()  # Reset the scene.

    def reset_base(self, model_id, pos, orn):
        """Resets the base position and orientation of the specified model."""
        self.physics_client.getBasePositionAndOrientation(model_id, pos, orn)

    def close(self):
        """Disconnects the physics client and closes the simulation."""
        self.physics_client.disconnect()

    def seed(self, seed=None, evaluate=False, validate=False):
        """
        Seeds the simulation's random number generator for reproducibility.
        
        Args:
            seed (int, optional): A seed for random number generation.
            evaluate (bool): If True, uses a fixed seed for evaluation.
            validate (bool): If True, sets a validation flag.
            
        Returns:
            RandomState: The random number generator.
        """
        if evaluate:
            self._validate = validate
