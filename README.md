# Deep-RL-Grasping



```
DeepRL-grasping/  
├── config/  
│   ├── gripper_grasp.yaml             # Configuration for full grasping experiments  
│   └── simplified_object_picking.yaml # Configuration for simplified object picking experiments  
│  
├── manipulation_main/  
│   ├── common/  
│   │   ├── io_utils.py                # Functions for configuration loading/saving, YAML parsing  
│   │   ├── transformations.py         # Functions for converting between Euler angles and quaternions  
│   │   ├── transform_utils.py         # Utility functions for handling transformation matrices  
│   │   └── camera_utils.py            # Utilities to create and process camera info and projection matrices  
│   │  
│   ├── gripperEnv/  
│   │   ├── robot.py                   # Defines the RobotEnv class (the main environment)  
│   │   │   └── Depends on common utilities, sensor, actuator, rewards, curriculum  
│   │   ├── sensor.py                  # Implements RGBDSensor and EncodedDepthImgSensor  
│   │   │   └── Uses common/io_utils, transform_utils, camera_utils, and encoders  
│   │   ├── actuator.py                # Implements Actuator to control robot actions  
│   │   ├── rewards.py                 # Defines various reward functions (Reward, SimplifiedReward, ShapedCustomReward)  
│   │   ├── curriculum.py              # Implements WorkspaceCurriculum for adaptive learning  
│   │   └── encoders.py                # Contains autoencoder models (e.g. SimpleAutoEncoder) for depth image encoding  
│   │  
│   ├── simulation/  
│   │   ├── simulation.py              # Defines the World class (simulation backbone using PyBullet)  
│   │   ├── scene.py                   # Contains concrete scene classes (e.g. OnTable, OnFloor)  
│   │   └── base_scene.py              # Abstract BaseScene class (provides common object sampling and scene configuration)  
│   │  
│   ├── training/  
│   │   ├── train_stable_baselines.py  # Main training and evaluation script; parses arguments and launches training  
│   │   │   └── Depends on sb_helper, wrapper, and common/io_utils  
│   │   ├── sb_helper.py               # Helper module that sets up and manages training using Stable‑Baselines3  
│   │   │   └── Depends on custom_obs_policy and base_callbacks  
│   │   ├── custom_obs_policy.py       # Defines custom CNN extractor (e.g. AugmentedNatureCNN) using PyTorch and SB3 torch_layers  
│   │   └── wrapper.py                 # Implements TimeFeatureWrapper to augment observations with a time feature  
│   │  
│   └── utils.py                       # Additional utility functions (e.g., run_agent for evaluation)  
│  
└── README.md                          # Project overview and instructions (if available)

```
