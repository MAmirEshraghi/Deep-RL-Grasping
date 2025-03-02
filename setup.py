from setuptools import setup, find_packages

setup(
    name='gripperEnv',
    version='0.0.1',
    packages=find_packages(
        include=['manipulation_main', 'manipulation_main.*'],
        exclude=['models', 'models.*', 'config', 'config.*']
    ),
    install_requires=[
        'stable-baselines3',
        'tensorflow==2.13.0',
        #'tensorflow_gpu=',
        #'tf-agents',
        'gym==0.26.2',
        #'keras==2.2.4',
        'matplotlib==3.7.2',
        'numpy==1.24',
        'opencv-contrib-python==4.8.0.76',
        'pandas==2.2',
        'pybullet==3.2.7',
        'pytest==7.4.2',
        'pydot==1.4.2',
        'PyYAML==6.0.1',
        'seaborn==0.12',
        'scikit-learn==1.3.2',
        'tqdm==4.66.1',
        'paramiko==2.12.0',
    ],
)
