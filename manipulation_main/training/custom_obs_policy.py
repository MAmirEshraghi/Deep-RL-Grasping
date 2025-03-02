import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class AugmentedNatureCNN(BaseFeaturesExtractor):
    """
    Custom CNN extractor that processes visual observations by applying convolutional layers
    to all channels except the last one (which contains direct features), and then concatenates
    a fixed number of direct features (either sliced or padded) with the CNN output.

    The final output is passed through a linear layer to obtain a compact feature representation.
    """
    def __init__(self, observation_space, num_direct_features, features_dim=512):
        """
        Initializes the feature extractor.

        :param observation_space: Gym space with shape either (channels, height, width) or (height, width, channels).
                                  The last channel is assumed to contain direct features.
        :param num_direct_features: Number of direct features to include from the last channel.
        :param features_dim: Desired output dimension.
        """
        super(AugmentedNatureCNN, self).__init__(observation_space, features_dim)
        self.num_direct_features = num_direct_features

        # Determine if observations are channels-first or channels-last.
        # We assume that if the first dimension is less than or equal to 4, it's channels-first.
        if observation_space.shape[0] <= 4:
            # channels-first: shape = (C, H, W)
            self.channels_first = True
            n_input_channels = observation_space.shape[0] - 1
            H, W = observation_space.shape[1], observation_space.shape[2]
        else:
            # channels-last: shape = (H, W, C)
            self.channels_first = False
            n_input_channels = observation_space.shape[-1] - 1
            H, W = observation_space.shape[0], observation_space.shape[1]

        # Determine kernel sizes and strides adaptively.
        kh1 = 8 if H >= 8 else H
        kw1 = 8 if W >= 8 else W
        stride1 = (4, 4) if W >= 8 else (4, 1)

        kh2 = 4 if H >= 4 else H
        kw2 = 4 if W >= 4 else W
        stride2 = (2, 2) if W >= 4 else (2, 1)

        kh3 = 3 if H >= 3 else H
        kw3 = 3 if W >= 3 else W
        stride3 = (1, 1)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=(kh1, kw1), stride=stride1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(kh2, kw2), stride=stride2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(kh3, kw3), stride=stride3),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the flattened output dimension using a dummy forward pass.
        with th.no_grad():
            sample_input = th.zeros((1, n_input_channels, H, W))
            n_flatten = self.cnn(sample_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(512 + num_direct_features, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        """
        Forward pass: process the CNN input and concatenate a fixed number of direct features.

        :param observations: Tensor with shape (batch_size, channels, height, width) or (batch_size, height, width, channels)
        :return: Extracted feature tensor.
        """
        # If observations are channels-last, convert them to channels-first.
        if not self.channels_first:
            observations = observations.permute(0, 3, 1, 2)

        # Split into CNN input (all channels except the last) and direct features (last channel).
        cnn_input = observations[:, :observations.shape[1]-1, :, :]
        direct_features = observations[:, -1, :, :]  # Shape: (batch_size, height, width)
        # Flatten direct features.
        direct_features = direct_features.view(direct_features.size(0), -1)
        # Ensure direct_features has exactly num_direct_features columns.
        if direct_features.size(1) > self.num_direct_features:
            direct_features = direct_features[:, :self.num_direct_features]
        elif direct_features.size(1) < self.num_direct_features:
            pad_size = self.num_direct_features - direct_features.size(1)
            direct_features = th.cat([direct_features, th.zeros(direct_features.size(0), pad_size, device=direct_features.device)], dim=1)
        cnn_output = self.cnn(cnn_input)
        cnn_output = self.fc(cnn_output)
        concatenated = th.cat([cnn_output, direct_features], dim=1)
        return self.linear(concatenated)

def create_augmented_nature_cnn(num_direct_features):
    """
    Returns a dict to be passed as policy_kwargs to a SB3 algorithm.
    It specifies the custom features extractor class and its parameters.

    :param num_direct_features: Number of direct features to extract from the observation.
    :return: Dictionary with keys 'features_extractor_class' and 'features_extractor_kwargs'.
    """
    return dict(
        features_extractor_class=AugmentedNatureCNN,
        features_extractor_kwargs=dict(num_direct_features=num_direct_features)
    )

