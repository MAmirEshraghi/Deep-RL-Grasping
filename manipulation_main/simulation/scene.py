"""
This script defines scene classes that set up different simulation environments.
Each scene (OnTable, OnFloor, and OnTableWithBox) extends BaseScene and provides a reset method
to load specific models (e.g., table, tray, plane) and sample random objects into the world.
These scenes are used by the simulation environment (World) to initialize the world with objects.
"""

import os  # For file path operations.
import pybullet as p  # PyBullet physics engine.
import numpy as np  # For numerical computations.
import pybullet_data  # Provides built-in data paths for PyBullet models.
from manipulation_main.simulation.base_scene import BaseScene  # Base class for scene setups.
from manipulation_main.common import transform_utils  # For random quaternion generation and other transforms.

class OnTable(BaseScene):
    """Scene configuration for a tabletop environment with a table, tray, plane, and random objects."""
    
    def reset(self):
        """Resets the scene by loading a plane, table, tray, and spawning random objects on the table."""
        self.table_path = 'table/table.urdf'  # Relative path to the table URDF.
        self.plane_path = 'plane.urdf'  # Relative path to the plane URDF.
        self._model_path = pybullet_data.getDataPath()  # Get the base path for PyBullet models.
        tray_path = os.path.join(self._model_path, 'tray/tray.urdf')  # Full path for the tray model.
        plane_urdf = os.path.join("models", self.plane_path)  # Construct URDF path for the plane.
        table_urdf = os.path.join("models", self.table_path)  # Construct URDF path for the table.
        # Add the plane to the world at a fixed position.
        self._world.add_model(plane_urdf, [0., 0., -1.], [0., 0., 0., 1.])
        # Add the table to the world at a specified height.
        self._world.add_model(table_urdf, [0., 0., -.82], [0., 0., 0., 1.])
        # Add the tray to the world with a scaling factor.
        self._world.add_model(tray_path, [0, 0.075, -0.19],
                              [0.0, 0.0, 1.0, 0.0], scaling=1.2)
        if self._rng is None:
            self._rng = np.random.RandomState()
        
        # Determine the number of random objects to sample.
        n_objects = self._rng.randint(self.min_objects, self.max_objects + 1)
        urdf_paths, scale = self._object_sampler(n_objects)  # Sample object URDF paths and scale.
        # Spawn each object at a random position with a random orientation.
        for path in urdf_paths:
            position = np.r_[self._rng.uniform(-self.extent, self.extent, 2), 0.1]
            orientation = transform_utils.random_quaternion(self._rng.rand(3))
            self._world.add_model(path, position, orientation, scaling=scale)
            self._world.run(0.4)  # Allow time for the object to settle.
        self._world.run(1.)  # Wait for the objects to come to rest.

class OnFloor(BaseScene):
    """Scene configuration for a floor environment with a plane and random objects."""
    
    def reset(self):
        """Resets the floor scene by loading a plane and spawning random objects on the floor."""
        self.plane_path = 'plane.urdf'  # Relative path to the plane URDF.
        plane_urdf = os.path.join("models", self.plane_path)  # Construct URDF path for the plane.
        self._world.add_model(plane_urdf, [0., 0., -0.196], [0., 0., 0., 1.])  # Add plane at a specific height.
        # Determine the number of random objects to sample.
        n_objects = self._rng.randint(self.min_objects, self.max_objects + 1)
        urdf_paths, scale = self._object_sampler(n_objects)  # Sample object URDF paths and scale.
        # Spawn each object at a random position with a random orientation.
        for path in urdf_paths:
            position = np.r_[self._rng.uniform(-self.extent, self.extent, 2), 0.1]
            orientation = transform_utils.random_quaternion(self._rng.rand(3))
            self._world.add_model(path, position, orientation, scaling=scale)
            self._world.run(0.4)  # Allow time for the object to settle.
        self._world.run(1.)  # Wait for the objects to come to rest.

class OnTableWithBox(BaseScene):
    """Scene configuration for a tabletop environment with an additional box, as used in Google Q-opt setups."""
    pass  # Not implemented; serves as a placeholder for future scene setups.
