"""
This script implements encoder models to learn compact representations of image observations.
It provides a base Encoder class with methods for training, testing, saving/loading weights, and plotting the network,
and a SimpleAutoEncoder that builds a vanilla convolutional autoencoder used for encoding depth images.
This module is used by sensor components (e.g., EncodedDepthImgSensor) to preprocess visual inputs.
Updated for TensorFlow 2.x.
"""

import os  # For handling file paths.
import numpy as np  # For numerical computations.
import tensorflow as tf  # TensorFlow 2.x backend for deep learning models.

# Use tf.keras imports for model building and training.
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping  # For managing training callbacks.
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, LeakyReLU, Reshape, UpSampling2D  # For constructing convolutional layers.
from tensorflow.keras.models import Model  # To create Keras models.
from tensorflow.keras.optimizers import Adam  # For model optimization.
from tensorflow.keras.utils import plot_model  # For visualizing the model architecture.

# Configure TensorFlow to allow dynamic GPU memory growth.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class Encoder(object):
    """Base class for learning abstract representations of image observations via autoencoding."""
    
    def __init__(self, config):
        """Initializes the encoder by setting up internal placeholders and building the network."""
        self._encoder = None
        self._decoder = None
        self._model = None
        self._build(config)  # Call the subclass-specific network construction method.

    def _build(self, config):
        """Abstract method to build the encoder/decoder network; must be implemented by subclasses."""
        raise NotImplementedError

    def load_weights(self, model_dir):
        """Loads pretrained weights from the specified directory."""
        model_dir = os.path.expanduser(model_dir)
        weights_path = os.path.join(model_dir, 'model.h5')
        self._model.load_weights(weights_path)

    def plot(self, model_dir):
        """Saves visualizations of the encoder and decoder network architectures to the specified directory."""
        model_dir = os.path.expanduser(model_dir)
        # Generate a diagram of the encoder and save it.
        plot_model(self._encoder, os.path.join(model_dir, 'encoder.png'),
                   show_shapes=True)
        # Generate a diagram of the decoder and save it.
        plot_model(self._decoder, os.path.join(model_dir, 'decoder.png'),
                   show_shapes=True)

    def train(self, inputs, targets, batch_size, epochs, model_dir):
        """Trains the autoencoder on the provided inputs and targets, logging training progress and saving the best model."""
        early_stopper = EarlyStopping(patience=25)  # Stop training if no improvement after 25 epochs.
        history_path = os.path.join(model_dir, 'history.csv')
        csv_logger = CSVLogger(history_path)  # Log training metrics to a CSV file.
        model_path = os.path.join(model_dir, 'model.h5')
        # Save the model's weights when validation loss improves.
        checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                     save_weights_only=True, save_best_only=True)
        # Fit the model on the training data with a 10% validation split.
        history = self._model.fit(inputs, targets, batch_size, epochs,
                                  validation_split=0.1, shuffle=True,
                                  callbacks=[csv_logger, checkpoint, early_stopper])
        return history.history

    def test(self, inputs, targets):
        """Evaluates the autoencoder on the given inputs and targets, returning the computed loss."""
        return self._model.evaluate(inputs, targets)

    def predict(self, imgs):
        """Generates predictions for the given images using the full autoencoder model."""
        return self._model.predict(imgs)

    def encode(self, imgs):
        """Generates encoded representations for the given images using only the encoder part of the network."""
        return self._encoder.predict(imgs)

    @property
    def encoding_shape(self):
        """Returns the shape of the encoded representation produced by the encoder."""
        return self._encoder.layers[-1].output_shape[1:]


class SimpleAutoEncoder(Encoder):
    """A vanilla autoencoder that learns a compact, low-dimensional representation of input images."""
    
    def _build(self, config):
        """Builds the convolutional autoencoder architecture based on the provided configuration parameters."""
        # Configure TensorFlow to allow dynamic GPU memory allocation (handled above globally).
        
        # Retrieve network architecture specifications and encoding dimension from config.
        network = config['network']  # List of dicts; each dict defines a convolutional layer.
        encoding_dim = config['encoding_dim']  # Size of the latent vector.
        alpha = config.get('alpha', 0.1)  # Negative slope coefficient for LeakyReLU activation.

        # Define the input layer with fixed shape (64x64 grayscale image).
        inputs = Input(shape=(64, 64, 1))

        # Build the encoder: apply successive convolutional layers.
        h = inputs
        for layer in network:
            # Convolve with specified number of filters, kernel size, and stride.
            h = Conv2D(filters=layer['filters'],
                       kernel_size=layer['kernel_size'],
                       strides=layer['strides'],
                       padding='same')(h)
            # Apply LeakyReLU activation to introduce non-linearity.
            h = LeakyReLU(alpha)(h)

        # Save the shape of the final convolutional output to reconstruct dimensions in the decoder.
        shape = h.shape[1:]
        h = Flatten()(h)  # Flatten the feature maps into a vector.
        h = Dense(encoding_dim)(h)  # Reduce to the desired encoding dimension.
        z = LeakyReLU(alpha)(h)  # Apply activation to the latent vector.

        # Define the encoder model mapping inputs to the latent representation.
        self._encoder = Model(inputs, z, name='encoder')

        # Build the decoder: reconstruct the image from the encoded representation.
        latent_inputs = Input(shape=(encoding_dim,))
        # First, project the latent vector back to the flattened feature map size.
        h = Dense(np.prod(shape))(latent_inputs)
        h = LeakyReLU(alpha)(h)
        # Reshape the vector back into the shape of the convolutional feature maps.
        h = Reshape(shape)(h)

        # Reverse the encoder layers to upsample back to the original image dimensions.
        for i in reversed(range(1, len(network))):
            # Upsample according to the stride used in the encoder.
            h = UpSampling2D(size=network[i]['strides'])(h)
            # Apply a convolution to reduce the number of filters as per the encoder's reversed order.
            h = Conv2D(filters=network[i - 1]['filters'],
                       kernel_size=network[i]['kernel_size'],
                       padding='same')(h)
            h = LeakyReLU(alpha)(h)

        # Final upsampling and convolution to produce the output image with a single channel.
        h = UpSampling2D(network[0]['strides'])(h)
        outputs = Conv2D(1, network[0]['kernel_size'], padding='same')(h)

        # Define the decoder model mapping latent inputs to reconstructed images.
        self._decoder = Model(latent_inputs, outputs, name='decoder')

        # Set the loss function and optimizer.
        loss = 'mean_squared_error'
        optimizer = Adam(lr=config['learning_rate'])

        # Connect the encoder and decoder to form the full autoencoder model.
        self._model = Model(inputs, self._decoder(self._encoder(inputs)))
        self._model.compile(optimizer=optimizer, loss=loss)
