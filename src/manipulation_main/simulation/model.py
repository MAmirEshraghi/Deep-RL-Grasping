"""
This script defines the Model class, which loads URDF or SDF files into the PyBullet simulation,
and provides methods to retrieve joint and base pose information.
It also defines helper classes (_Link and _Joint) for interacting with individual links and joints of a model.
These classes are used by the simulation environment (World) to manage robot and object models.
"""
import pybullet as p  # Import PyBullet for physics simulation.
import numpy as np  # For numerical operations.

class Model(object):
    """Loads a simulation model (URDF/SDF) into PyBullet and organizes its joints and links."""
    
    def __init__(self, physics_client):
        """Initializes the Model with a reference to the PyBullet physics client."""
        self._physics_client = physics_client

    def load_model(self, path, start_pos=[0, 0, 0], 
                   start_orn=[0, 0, 0, 1], scaling=1., static=False):
        """Loads the model from a file (SDF or URDF), sets its initial pose, and builds joint/link objects."""
        if path.endswith('.sdf'):
            # Load an SDF model and reset its base pose.
            model_id = self._physics_client.loadSDF(path, globalScaling=scaling)[0]
            self._physics_client.resetBasePositionAndOrientation(model_id, start_pos, start_orn)
        else:
            # Load a URDF model with specified scaling and static flag.
            model_id = self._physics_client.loadURDF(
                path, start_pos, start_orn,
                globalScaling=scaling, useFixedBase=static)
        self.model_id = model_id
        # Build dictionaries for joints and links.
        joints, links = {}, {}
        for i in range(self._physics_client.getNumJoints(self.model_id)):
            joint_info = self._physics_client.getJointInfo(self.model_id, i)
            # Store joint limits (lower, upper, force) for each joint.
            joint_limits = {'lower': joint_info[8], 'upper': joint_info[9],
                            'force': joint_info[10]}
            joints[i] = _Joint(self._physics_client, self.model_id, i, joint_limits)
            links[i] = _Link(self._physics_client, self.model_id, i)
        self.joints, self.links = joints, links
        return model_id

    def get_joints(self):
        """Updates the joints dictionary with current joint information from the simulation."""
        for i in range(self._physics_client.getNumJoints(self.model_id)):
            self.joints[i] = self._physics_client.getJointInfo(self.model_id, i)

    def get_pose(self):
        """Returns the pose (position and orientation) of a specific link (link index 3) of the model."""
        pos, orn, _, _, _, _ = self._physics_client.getLinkState(self.model_id, 3)
        return (pos, orn)
    
    def getBase(self):
        """Returns the base position and orientation of the model."""
        return self._physics_client.getBasePositionAndOrientation(self.model_id)

class _Link(object):
    """Represents a single link of a model loaded into the simulation."""
    
    def __init__(self, physics_client, model_id, link_id):
        """Initializes a _Link with its physics client, model identifier, and link index."""
        self._physics_client = physics_client
        self.model_id = model_id
        self.lid = link_id

    def get_pose(self):
        """Returns the pose (position and orientation) of this link."""
        link_state = p.getLinkState(self.model_id, self.lid)
        position, orientation = link_state[0], link_state[1]
        return position, orientation

class _Joint(object):
    """Represents a single joint of a model, providing methods to read and control its position."""
    
    def __init__(self, physics_client, model_id, joint_id, limits):
        """Initializes a _Joint with its physics client, model identifier, joint index, and limits."""
        self._physics_client = physics_client
        self.model_id = model_id
        self.jid = joint_id
        self.limits = limits

    def get_position(self):
        """Returns the current position (angle/translation) of the joint."""
        joint_state = self._physics_client.getJointState(self.model_id, self.jid)
        return joint_state[0]

    def set_position(self, position, max_force=100.):
        """Sets the joint to a target position using position control with a specified maximum force."""
        self._physics_client.setJointMotorControl2(
            self.model_id, self.jid,
            controlMode=p.POSITION_CONTROL,
            targetPosition=position,
            force=max_force)

    def disable_motor(self):
        """Disables the joint's motor by setting its force to zero in velocity control mode."""
        self._physics_client.setJointMotorControl2(
            self.model_id, self.jid, controlMode=p.VELOCITY_CONTROL, force=0.)
