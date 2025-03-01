"""
This script trains and evaluates an autoencoder for depth image observations.
It loads configuration and dataset files, preprocesses depth images, trains the autoencoder
using the SimpleAutoEncoder defined in encoders.py, and provides options to test and visualize
the reconstruction performance. This script is meant to be run separately from the RL training.
"""

import argparse  # For command-line argument parsing.
import os  # For filesystem operations.
import pickle  # For loading serialized dataset files.
import matplotlib.pyplot as plt  # For plotting images.
from mpl_toolkits.axes_grid1 import ImageGrid  # For creating image grids in plots.
import numpy as np  # For numerical operations.

# Import utility functions for loading YAML configuration files.
from manipulation_main.common import io_utils
# Import the autoencoder definitions.
from manipulation_main.gripperEnv import encoders


def _load_data_set(data_path, test):
    """Loads the dataset from a pickle file and returns the test or training split based on the 'test' flag."""
    with open(os.path.expanduser(data_path), 'rb') as f:
        dataset = pickle.load(f)
    return dataset['test'] if test else dataset['train']


def _preprocess_depth(data_set):
    """Preprocesses depth images by removing flat surfaces and gripper pixels based on the associated masks."""
    depth_imgs = data_set['depth']
    masks = data_set['masks']
    for i in range(depth_imgs.shape[0]):
        img, mask = depth_imgs[i].squeeze(), masks[i].squeeze()
        img[mask == 0] = 0.  # Remove flat surfaces.
        img[mask == np.max(mask)] = 0.  # Remove the gripper from the depth image.
        depth_imgs[i, :, :, 0] = img
    return depth_imgs


def train(args):
    """Trains the autoencoder using the training dataset and saves the trained model along with the configuration."""
    # Load the encoder configuration from a YAML file.
    config = io_utils.load_yaml(args.config)
    
    # Ensure the model directory exists, creating it if necessary.
    model_dir = os.path.expanduser(args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    
    # Build the autoencoder model using the SimpleAutoEncoder class.
    model = encoders.SimpleAutoEncoder(config)
    # Save the current configuration for reproducibility.
    io_utils.save_yaml(config, os.path.join(model_dir, 'config.yaml'))
    
    # Load and preprocess the training dataset.
    train_set = _load_data_set(config['data_path'], test=False)
    train_imgs = _preprocess_depth(train_set)
    
    # Retrieve training hyperparameters and start the training process.
    batch_size = config['batch_size']
    epochs = config['epochs']
    model.train(train_imgs, train_imgs, batch_size, epochs, model_dir)


def test(args):
    """Tests the trained autoencoder on the test dataset and prints the computed loss."""
    # Load the autoencoder configuration.
    config = io_utils.load_yaml(os.path.join(args.model_dir, 'config.yaml'))
    # Build the model and load pretrained weights.
    model = encoders.SimpleAutoEncoder(config)
    model.load_weights(args.model_dir)
    
    # Load and preprocess the test dataset.
    test_set = _load_data_set(config['data_path'], test=True)
    test_imgs = _preprocess_depth(test_set)
    
    # Evaluate the model on the test set and print the loss.
    loss = model.test(test_imgs, test_imgs)
    print('Test loss: {}'.format(loss))
    return loss


def plot_history(args):
    """Placeholder function for plotting training history (to be implemented later)."""
    pass
    # TODO: implement plotting of training history using a utility function.


def visualize(args):
    """Visualizes reconstructed depth images by comparing original, reconstruction, and error images."""
    n_imgs = 2  # Number of images to visualize.
    
    # Load the autoencoder configuration and model weights.
    config = io_utils.load_yaml(os.path.join(args.model_dir, 'config.yaml'))
    model = encoders.SimpleAutoEncoder(config)
    model.load_weights(args.model_dir)
    
    # Load a random selection of test images from the dataset.
    test_set = _load_data_set(config['data_path'], test=True)
    selection = np.random.choice(test_set['rgb'].shape[0], size=n_imgs)
    rgb = test_set['rgb'][selection]
    depth = _preprocess_depth(test_set)[selection]
    
    # Reconstruct depth images and compute absolute errors.
    reconstruction = model.predict(depth)
    error = np.abs(depth - reconstruction)
    
    # Create an image grid for visualization.
    fig = plt.figure()
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(n_imgs, 4),
                     share_all=True,
                     axes_pad=0.05,
                     cbar_mode='single',
                     cbar_location='right',
                     cbar_size='5%',
                     cbar_pad=None)
    
    def _plot_sample(i, rgb, depth, reconstruction, error):
        """Plots a single sample's RGB, depth, reconstruction, and error images in the grid."""
        ax = grid[4 * i]
        ax.set_axis_off()
        ax.imshow(rgb)  # Show the RGB image.
        
        def _add_depth_img(depth_img, j):
            ax = grid[4 * i + j]
            ax.set_axis_off()
            img = ax.imshow(depth_img.squeeze(), cmap='viridis')
            img.set_clim(0., 0.3)
            ax.cax.colorbar(img)
        
        # Plot depth, reconstructed image, and error image.
        _add_depth_img(depth, 1)
        _add_depth_img(reconstruction, 2)
        _add_depth_img(error, 3)
    
    # Loop through the selected images and plot each sample.
    for i in range(n_imgs):
        _plot_sample(i, rgb[i], depth[i], reconstruction[i], error[i])
    
    # Save the visualization to a file and display it.
    plt.savefig(os.path.join(args.model_dir, 'reconstructions.png'), dpi=300)
    plt.show()


if __name__ == '__main__':
    """Parses command-line arguments and executes the corresponding function (train, test, plot_history, or visualize)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    
    subparsers = parser.add_subparsers()
    
    # Define sub-command for training the autoencoder.
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--config', type=str, required=True)
    train_parser.set_defaults(func=train)
    
    # Define sub-command for testing the autoencoder.
    test_parser = subparsers.add_parser('test')
    test_parser.set_defaults(func=test)
    
    # Define sub-command for plotting the training history.
    plot_parser = subparsers.add_parser('plot_history')
    plot_parser.set_defaults(func=plot_history)
    
    # Define sub-command for visualizing reconstructed images.
    vis_parser = subparsers.add_parser('visualize')
    vis_parser.set_defaults(func=visualize)
    
    args = parser.parse_args()
    args.func(args)
