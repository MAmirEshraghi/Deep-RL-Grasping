
# Deep-RL-Grasping 
## (in progress)

DeepRL-Grasping is a modular repository for training reinforcement learning agents to perform robotic grasping in simulated environments using PyBullet and Stable-Baselines3. The project is designed with a clear separation of concerns, including modules for common utilities, simulation, environment definitions, sensors, actuators, rewards, curriculum learning, and training scripts.

![07](https://github.com/user-attachments/assets/dc315482-a9cf-487c-a1db-2b4566d74de3)

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

# Installation and Usage

## Installation

### Clone the Repository
```bash
git clone https://github.com/MAmirEshraghi/DeepRL-Grasping.git
cd DeepRL-Grasping
```


## Usage
# Training
To train an agent using SAC with a specific configuration, run:

```bash
python manipulation_main/training/train_stable_baselines.py train --config config/simplified_object_picking.yaml --algo SAC --model_dir trained_models/SAC_full --timestep 10000 -v
```

train: Calls the training sub-command.
* --config: Path to the YAML configuration file.
* --algo: RL algorithm (e.g., SAC, PPO).
* --model_dir: Directory for saving the model and logs.
* --timestep: Total number of timesteps for training.
* -v: Enables visualization.

# Evaluation
To run a trained model, execute:

```bash
python manipulation_main/training/train_stable_baselines.py run --model trained_models/SAC_full/best_model/best_model.zip -v -t -s
```

run: Calls the evaluation sub-command.
* --model: Path to the saved model file.
* -v: Enables visualization during evaluation.
* -t: Runs on the test dataset.
* -s: Uses stochastic actions during execution.

# Configuration
## YAML Files:
The config folder contains YAML files that define parameters for different tasks and environments (e.g., gripper_grasp.yaml, simplified_object_picking.yaml).
Dependencies
* Python 3.11
* PyBullet 3.2.7
* Gymnasium (formerly Gym)
* Stable-Baselines3
* PyTorch (used by Stable-Baselines3)
* OpenCV 4.8.0.76
* NumPy 

* **Domain and Scene Transfer**
![01](https://github.com/user-attachments/assets/0c935520-b7d5-467d-b41e-5c4039e7104d)


* **Different Perception Layers**
![02](https://github.com/user-attachments/assets/cbe1afa3-d453-4132-b3c4-68d89448705b)


* **Training Environment**
![04](https://github.com/user-attachments/assets/c99e80d1-86e5-4f06-9095-b644a348e51c)

  
* **Domain transfer performance**
![06](https://github.com/user-attachments/assets/2d6c959c-0e68-477c-8d37-2837963dec1e)

![SACGripperEnvRes](https://github.com/user-attachments/assets/b98521ac-7653-4894-abac-a447b3ee057b)


# Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

# License
MIT License
