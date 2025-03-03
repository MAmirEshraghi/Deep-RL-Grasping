
# Deep-RL-Grasping

DeepRL-Grasping is a modular repository for training reinforcement learning agents to perform robotic grasping in simulated environments using PyBullet and Stable-Baselines3. The project is designed with a clear separation of concerns, including modules for common utilities, simulation, environment definitions, sensors, actuators, rewards, curriculum learning, and training scripts.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview

DeepRL-Grasping provides a complete framework for training RL agents to solve grasping tasks. The repository includes:

- **Common Utilities:** Functions for I/O, configuration management, coordinate transformations, and camera calculations.
- **Simulation:** A PyBullet-based simulation engine, including scene setup and object sampling.
- **Gripper Environment:** The `RobotEnv` class that integrates sensors (e.g., RGB-D and encoded depth) and actuators to simulate a robotic gripper.
- **Rewards and Curriculum:** Modules that define reward functions and curriculum strategies to improve learning.
- **Training:** Scripts that use Stable-Baselines3 to train and evaluate agents using various RL algorithms (e.g., SAC, PPO, DQN).

## Repository Structure

```plaintext
DeepRL-Grasping/
├── config/
│   ├── gripper_grasp.yaml             # Full grasping configuration.
│   └── simplified_object_picking.yaml # Simplified object picking configuration.
│
├── manipulation_main/
│   ├── common/  
│   │   ├── io_utils.py                # YAML loading/saving and other utilities.
│   │   ├── transformations.py         # Conversion between Euler angles and quaternions.
│   │   ├── transform_utils.py         # Additional transformation utilities.
│   │   └── camera_utils.py            # Camera information and projection matrix helpers.
│   │  
│   ├── gripperEnv/
│   │   ├── robot.py                   # The main environment class (RobotEnv).
│   │   ├── sensor.py                  # Sensor implementations (RGBDSensor, EncodedDepthImgSensor).
│   │   ├── actuator.py                # Actuator control for the robot.
│   │   ├── rewards.py                 # Reward functions (Reward, SimplifiedReward, ShapedCustomReward).
│   │   ├── curriculum.py              # Workspace curriculum management.
│   │   └── encoders.py                # Autoencoder models for image encoding.
│   │  
│   ├── simulation/
│   │   ├── simulation.py              # World class for simulation using PyBullet.
│   │   ├── scene.py                   # Concrete scene classes (e.g., OnTable, OnFloor).
│   │   └── base_scene.py              # Abstract BaseScene class.
│   │  
│   ├── training/
│   │   ├── train_stable_baselines.py  # Main training and evaluation script.
│   │   ├── sb_helper.py               # Training helper for Stable-Baselines3.
│   │   ├── custom_obs_policy.py       # Custom CNN feature extractor for policy networks.
│   │   └── wrapper.py                 # Environment wrappers (e.g., TimeFeatureWrapper).
│   │  
│   └── utils.py                       # Additional utility functions (e.g., run_agent).
│
└── README.md                          # Project overview and instructions.

│   │   └── wrapper.py                 # Implements TimeFeatureWrapper to augment observations with a time feature  
│   │  
│   └── utils.py                       # Additional utility functions (e.g., run_agent for evaluation)  
│  
└── README.md                          # Project overview and instructions (if available)

```
