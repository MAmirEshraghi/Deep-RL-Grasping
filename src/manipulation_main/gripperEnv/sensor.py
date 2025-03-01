"""
This script implements sensor functionality for the robotic simulation by defining classes that capture
and process synthetic RGB-D images using PyBullet's renderer and, optionally, encode depth images with a pretrained autoencoder.
"""

import copy   # For deep copying camera configuration data.
import os

import cv2    # OpenCV for potential visualization.
import gym
import numpy as np
import pybullet as p
import tensorflow as tf

# Import utility modules for file I/O, transformations, and camera-specific functions.
from manipulation_main.common import io_utils, transform_utils, camera_utils
# Import encoder definitions for encoding depth images.
from manipulation_main.gripperEnv import encoders


class RGBDSensor:
    """Collects synthetic RGB-D images from the scene and optionally randomizes camera parameters."""
    
    def __init__(self, config, robot, randomize=True):
        """Initializes the RGB-D sensor with intrinsics, extrinsics, and a camera model based on the given configuration."""
        self._physics_client = robot.physics_client  # Reference to the physics client used for rendering.
        self._robot = robot  # Reference to the robot environment.
        full_obs = config.get('full_observation', False)  # Flag indicating if full (RGB+Depth) observation is used.
        intrinsics_path = config['camera_info']  # Path to the camera intrinsics YAML file.
        extrinsics_path = config['transform']  # Path to the camera extrinsics YAML file.
        extrinsics_dict = io_utils.load_yaml(extrinsics_path)  # Load extrinsic transformation parameters.
        
        self._camera_info = io_utils.load_yaml(intrinsics_path)  # Load camera intrinsic parameters.
        self._transform = transform_utils.from_dict(extrinsics_dict)  # Convert extrinsics dictionary to a transformation matrix.
        
        self._randomize = config.get('randomize', None) if randomize else None  # Set randomization parameters if enabled.
        self._construct_camera(self._camera_info, self._transform)  # Construct the camera using the provided info and transform.
        
        # Define the sensor state space as a Box; if full observation, include additional channels.
        self.state_space = gym.spaces.Box(low=0, high=1,
                shape=(self.camera.info.height, self.camera.info.width, 1))
        if full_obs:
            # For full observation, the state space includes RGB + Depth (5 channels).
            self.state_space = gym.spaces.Box(low=0, high=255,
                shape=(self.camera.info.height, self.camera.info.width, 5))
    
    def reset(self):
        """Randomizes the camera parameters if randomization is enabled and re-constructs the camera."""
        if self._randomize is None:
            return
        
        camera_info = copy.deepcopy(self._camera_info)  # Create a deep copy of the original camera info.
        transform = np.copy(self._transform)  # Copy the original transformation matrix.
        
        # Extract randomization ranges from the configuration.
        f = self._randomize['focal_length']
        c = self._randomize['optical_center']
        t = self._randomize['translation']
        r = self._randomize['rotation']
        
        # Randomize the focal lengths (fx and fy).
        camera_info['K'][0] += np.random.uniform(-f, f)
        camera_info['K'][4] += np.random.uniform(-f, f)
        # Randomize the optical center (cx and cy).
        camera_info['K'][2] += np.random.uniform(-c, c)
        camera_info['K'][5] += np.random.uniform(-c, c)
        # Randomize the translation component.
        magnitue = np.random.uniform(0., t)
        direction = transform_utils.random_unit_vector()
        transform[:3, 3] += magnitue * direction
        # Randomize the rotation using a random axis and angle.
        angle = np.random.uniform(0., r)
        axis = transform_utils.random_unit_vector()
        q = transform_utils.quaternion_about_axis(angle, axis)
        transform = np.dot(transform_utils.quaternion_matrix(q), transform)
        
        self._construct_camera(camera_info, transform)  # Rebuild the camera with the new randomized parameters.
    
    def get_state(self):
        """Renders and returns an RGB image, a depth image, and a segmentation mask from the current viewpoint."""
        h_world_robot = transform_utils.from_pose(*self._robot.get_pose())  # Get the current world-to-robot transform.
        h_camera_world = np.linalg.inv(np.dot(h_world_robot, self._h_robot_camera))  # Compute the camera-to-world transform.
        rgb, depth, mask = self.camera.render_images(h_camera_world)  # Render the images using the camera model.
        return rgb, depth, mask
    
    def _construct_camera(self, camera_info, transform):
        """Constructs the RGBDCamera using the provided camera information and transformation."""
        self.camera = RGBDCamera(self._physics_client, camera_info)  # Create a new camera instance.
        self._h_robot_camera = transform  # Store the transformation from robot to camera.

        
class RGBDCamera(object):
    """Implements an OpenCV-compliant camera model using PyBullet's renderer."""
    
    def __init__(self, physics_client, config):
        """Initializes the camera with intrinsics, near/far planes, and computes its projection matrix."""
        self._physics_client = physics_client  # Reference to the physics client.
        self.info = camera_utils.CameraInfo.from_dict(config)  # Load camera intrinsic parameters into a CameraInfo object.
        self._near = config['near']  # Near clipping plane.
        self._far = config['far']    # Far clipping plane.
        
        # Build the projection matrix using the camera's intrinsics and clipping planes.
        self.projection_matrix = _build_projection_matrix(
            self.info.height, self.info.width, self.info.K, self._near, self._far)
    
    def render_images(self, view_matrix):
        """Renders RGB, depth, and segmentation mask images from the given view matrix."""
        gl_view_matrix = view_matrix.copy()
        gl_view_matrix[2, :] *= -1  # Flip the Z axis for OpenGL compatibility.
        gl_view_matrix = gl_view_matrix.flatten(order='F')  # Flatten the view matrix in column-major order.
        
        gl_projection_matrix = self.projection_matrix.flatten(order='F')  # Flatten the projection matrix.
        
        # Request the camera image from PyBullet.
        result = self._physics_client.getCameraImage(
            width=self.info.width,
            height=self.info.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_projection_matrix,
            renderer=p.ER_TINY_RENDERER)
        
        # Process the returned data to extract the RGB image.
        rgb = np.asarray(result[2], dtype=np.uint8)
        rgb = np.reshape(rgb, (self.info.height, self.info.width, 4))[:, :, :3]
        # Process the depth buffer to compute a depth image.
        near, far = self._near, self._far
        depth_buffer = np.asarray(result[3], np.float32).reshape(
            (self.info.height, self.info.width))
        depth = 1. * far * near / (far - (far - near) * depth_buffer)
        
        # Extract the segmentation mask.
        mask = result[4]
        
        return rgb, depth, mask


def _gl_ortho(left, right, bottom, top, near, far):
    """Computes an orthographic projection matrix similar to OpenGL's glOrtho."""
    ortho = np.diag([2./(right-left), 2./(top-bottom), -2./(far-near), 1.])
    ortho[0, 3] = - (right + left) / (right - left)
    ortho[1, 3] = - (top + bottom) / (top - bottom)
    ortho[2, 3] = - (far + near) / (far - near)
    return ortho


def _build_projection_matrix(height, width, K, near, far):
    """Builds the combined projection matrix using the camera's intrinsic matrix and an orthographic projection."""
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    perspective = np.array([[fx, 0., -cx, 0.],
                            [0., fy, -cy, 0.],
                            [0., 0., near + far, near * far],
                            [0., 0., -1., 0.]])
    ortho = _gl_ortho(0., width, height, 0., near, far)
    return np.matmul(ortho, perspective)


class EncodedDepthImgSensor:
    """Encodes depth images from a sensor using a pretrained autoencoder network."""
    
    def __init__(self, config, sensor, robot):
        """Initializes the encoder by building its network, loading pretrained weights, and setting the state space."""
        self.scope = 'encoded_img_sensor'  # TensorFlow scope name.
        self._sensor = sensor  # Reference to the original depth sensor.
        self._robot = robot  # Reference to the robot environment.
        self.scene_type = config['scene'].get('scene_type', "OnTable")  # Scene type to adjust filtering.
        config = config['sensor']
        self._visualize = config.get('visualize', False)  # Flag to enable visualization.
        
        # Load the encoder configuration and weights from the specified directory.
        model_dir = config['encoder_dir']
        encoder_config = io_utils.load_yaml(os.path.join(model_dir, 'config.yaml'))
        
        # Build the encoder network and load its pretrained weights.
        with tf.name_scope(self.scope):
            self._encoder = encoders.SimpleAutoEncoder(encoder_config)
            self._encoder.load_weights(model_dir)
        
        # Define the state space for the encoded depth image.
        dim = int(np.prod(self._encoder.encoding_shape))
        self.state_space = gym.spaces.Box(-1., 1., (dim,), np.float32)
        
        # Optionally create an OpenCV window for visualization.
        if self._visualize:
            cv2.namedWindow('imgs', flags=cv2.WINDOW_NORMAL)
    
    def get_variables(self):
        """Returns a list of TensorFlow variables associated with the encoder network."""
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    
    def get_state(self):
        """Encodes the depth image from the current viewpoint after filtering out unwanted regions."""
        # Render the depth image and segmentation mask from the sensor.
        _, img, mask = self._sensor.get_state()
        
        # Filter out pixels corresponding to the plane and robot from the image.
        img[mask == 0] = 0.
        img[mask == self._robot.robot_id] = 0.
        # If the scene type is "OnTable", filter out additional objects.
        if self.scene_type == "OnTable":
            img[mask == 1] = 0.
            img[mask == 2] = 0.
        
        # Reshape the image to prepare it for the encoder.
        height, width = img.shape
        input_img = np.reshape(img, (1, height, width, 1))
        # Compute the encoding of the input image.
        encoding = self._encoder.encode(input_img).squeeze()
        if self._visualize:
            # If visualization is enabled, reconstruct the image and display the error.
            reconstructed_img = np.squeeze(self._encoder.predict(input_img)[0])
            error_img = np.abs(img - reconstructed_img)
            stacked_imgs = np.vstack((img, reconstructed_img, error_img))
            cv2.imshow('imgs', 4. * stacked_imgs)
            cv2.waitKey(1)
        
        return encoding
