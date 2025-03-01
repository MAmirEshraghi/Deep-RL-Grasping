"""
This script defines the abstract BaseScene class, which serves as a template for setting up different
simulation scenes by sampling objects and configuring scene parameters. It provides common functionality
for sampling object URDF paths from predefined datasets, and its reset() method must be implemented by subclasses.
"""
import os  # For file and path operations.
import numpy as np  # For numerical operations.
import pybullet as p  # For physics simulation functions.
import pybullet_data  # Provides access to default PyBullet data paths.
from abc import ABC, abstractmethod  # For defining an abstract base class.

class BaseScene(ABC):
    """Abstract base class for defining simulation scenes with object sampling and configuration."""
    
    def __init__(self, world, config, rng, test=False, validate=False):
        """Initializes the base scene with world reference, configuration, RNG, and mode flags."""
        self._world = world  # Reference to the simulation world.
        self._rng = rng  # Random number generator for sampling.
        self._model_path = pybullet_data.getDataPath()  # Set the default model path from PyBullet data.
        self._validate = validate  # Flag indicating validation mode.
        self._test = test  # Flag indicating test mode.
        self.extent = config.get('extent', 0.1)  # Scene extent range.
        self.max_objects = config.get('max_objects', 6)  # Maximum number of objects to spawn.
        self.min_objects = config.get('min_objects', 1)  # Minimum number of objects to spawn.
        # Map dataset names to their corresponding object sampler methods.
        object_samplers = {'wooden_blocks': self._sample_wooden_blocks,
                           'random_urdfs': self._sample_random_objects}
        # Select the object sampler based on configuration.
        self._object_sampler = object_samplers[config['scene']['data_set']]
        print("dataset", config['scene']['data_set'])  # Output the selected dataset.

    def _sample_wooden_blocks(self, n_objects):
        """Samples wooden block URDF paths from a fixed list and returns them with a default scaling factor."""
        self._model_path = "models/"  # Override the model path to local models.
        object_names = ['circular_segment', 'cube',
                        'cuboid0', 'cuboid1', 'cylinder', 'triangle']  # List of available wooden block names.
        selection = self._rng.choice(object_names, size=n_objects)  # Randomly select a set of object names.
        paths = [os.path.join(self._model_path, 'wooden_blocks',
                              name + '.urdf') for name in selection]  # Build full URDF paths for the selected objects.
        return paths, 1.  # Return the paths with a default scaling factor of 1.

    def _sample_random_objects(self, n_objects):
        """Samples random object URDF paths based on a defined range and mode, returning them with a default scaling."""
        if self._validate:
            self.object_range = np.arange(700, 850)  # Use a fixed range for validation.
        elif self._test:
            self.object_range = np.arange(850, 1000)  # Use a different range for testing.
        else: 
            self.object_range = 700  # Use a constant value if not in test or validate mode.
        selection = self._rng.choice(self.object_range, size=n_objects)  # Randomly select object IDs from the range.
        paths = [os.path.join(self._model_path, 'random_urdfs',
                            '{0:03d}/{0:03d}.urdf'.format(i)) for i in selection]  # Construct the URDF paths.
        return paths, 1.  # Return the paths with a default scaling factor of 1.

    @abstractmethod
    def reset(self):
        """Abstract method that must be implemented to reset the scene and initialize models."""
        raise NotImplementedError
