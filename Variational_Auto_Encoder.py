# 1st Party Modules
from os import getcwd
from os.path import join

# 3rd Party Modules
import numpy as np

import tensorflow as tf 
from tensorflow.keras.datasets.mnist import load_data

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["figure.figsize"] = (12.8, 7.20)

# Custom Modules
from assignment4_part1 import Helper


class Decoder(tf.keras.Model):
    """ Buidling the Decoder Part of the model. """

    def __init__(self, latent_dim: int):
        super(Decoder, self).__init__()

        self.dense = tf.keras.layers.Dense(7 * 7 * 64, activation=tf.nn.relu)
        self.reshape = tf.keras.layers.Reshape((7, 7, 64))

        self.conv_1 = tf.keras.layers.Conv2DTranspose(64, (3,3), strides=2,
            activation=tf.nn.relu, padding="same")

        self.conv_2 = tf.keras.layers.Conv2DTranspose(32, (3,3), strides=2, 
            activation=tf.nn.relu, padding="same")

        self.decoder_outputs = tf.keras.layers.Conv2DTranspose(1, 3, 
            activation=tf.nn.sigmoid, padding="same")


    def call(self, inputs):
        out_dense = self.dense(inputs)
        out_dense = self.reshape(out_dense)

        out_conv1 = self.conv_1(out_dense)
        out_conv2 = self.conv_2(out_conv1)

        out_decoder = self.decoder_outputs(out_conv2)

        return out_decoder


class Encoder(tf.keras.Model):
    """ Building the Encoder Part for the VAE. """

    def __init__(self, latent_dim: int):
        super(Encoder, self).__init__()
        
        # Convolutional layer
        self.conv_1 = tf.keras.layers.Conv2D(32, (3,3), strides=2, 
            activation=tf.nn.relu, padding="same")

        self.conv_2 = tf.keras.layers.Conv2D(64, (3,3), strides=2,
            activation=tf.nn.relu, padding="same")

        # Flatten
        self.flatten = tf.keras.layers.Flatten(data_format="channels_last")
        
        # Dense layers
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)
        self.z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")

        # Sampling layer
        self.z = SamplingLayer()


    def call(self, inputs):
        out_conv1 = self.conv_1(inputs)
        out_conv2 = self.conv_2(out_conv1)

        out_flat = self.flatten(out_conv2)
        out_dense = self.dense(out_flat)

        out_mean = self.z_mean(out_dense)
        out_lvar = self.z_log_var(out_dense)

        out_z = self.z([out_mean, out_lvar])
        
        return [out_mean, out_lvar, out_z]


class SamplingLayer(tf.keras.layers.Layer):
    """ Sampling Layer class to sample the vector z,
    which encodes a digit.

    This code is directly taken respectively adapted from 
    https://keras.io/examples/generative/vae/ (21.01.21).

    Author: fchollet
    Date created: 2020/05/03
    Last modified: 2020/05/03
    Description: Convolutional Variational AutoEncoder (VAE) 
        trained on MNIST digits.
    """

    def call(self, inputs):

        mean_z, z_log_var = inputs

        batch_size = tf.shape(mean_z)[0]
        dim_size = tf.shape(mean_z)[1]

        epsilon = tf.keras.backend.random_normal(shape=(batch_size, dim_size))

        return mean_z + tf.exp(0.5 * z_log_var) * epsilon


class VAE(tf.keras.Model):
    """ Building the variational auto encoder itself. """

    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_track = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_track = tf.keras.metrics.Mean(name="recon_loss")
        self.kldiv_loss_track = tf.keras.metrics.Mean(name="kldiv_loss")

    
    @property
    def metrics(self):
        """ Getter for metrics list. """

        return [
            self.total_loss_track, 
            self.recon_loss_track, 
            self.kldiv_loss_track
            ]

    # Apparently data is of type
    # Tensor("IteratorGetNext:0", shape=(None, 28, 28, 1), dtype=float32)
    # which might be the culprit of the error
    #
    # OperatorNotAllowedInGraphError: iterating over `tf.Tensor` is not 
    # allowed: AutoGraph did convert this function. This might indicate 
    # you are trying to use an unsupported feature.
    #
    # It seems like, I try to iterate of a Tensor object, which is 
    # unfortunately not 
    # supported.
    # 
    # Trying to convert data to a numpy array via data.numpy results 
    # in an AttributeError
    def train_step(self, data):
        with tf.GradientTape() as tape:

            z_mean, z_log_var, z = self.encoder(data)
            recon = self.decoder(z)

            bin_cross_entropy = tf.keras.losses.binary_crossentropy(data,recon)
            bin_cross_entropy = tf.reduce_sum(bin_cross_entropy, axis=(1, 2)) 

            recon_loss = tf.reduce_mean(bin_cross_entropy)
            kldiv_loss = -0.5*(1+z_log_var-tf.square(z_mean)-tf.exp(z_log_var))
            kldiv_loss = tf.reduce_mean(tf.reduce_sum(kldiv_loss, axis=1))
            total_loss = recon_loss + kldiv_loss

        gradients = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.total_loss_track.update_state(total_loss)
        self.recon_loss_track.update_state(recon_loss)
        self.kldiv_loss_track.update_state(kldiv_loss)

        # Configure loss dict
        loss_dict = {
            "total_loss": self.total_loss_track.result(),
            "recon_loss": self.recon_loss_track.result(),
            "kldiv_loss": self.kldiv_loss_track.result()
        }

        return loss_dict


        @staticmethod
        def recon_images(pred_samples, num_images: int = 12, num_cols: int = 10):
            """ Grid of sampled digits.
            
            Adapted from:
            https://keras.io/examples/generative/vae/
            
            """
            
            # Params for plot 
            digit_size = 28
            scale_fact = 1.0
            fig = np.zeros((digit_size * num_images, igit_size * num_cols))

            # lin. spaced coordinates of 2D-Plot
            x_grid = np.linspace(-scale_fact, scale_fact, num_images)
            y_grid = np.linspace(-scale_fact, scale_fact, num_cols)

            # iterate over grid


if __name__ == "__main__":
    
    # Latent dimensions of VAE        
    latent_dim=2

    decoder = Decoder(latent_dim)
    encoder = Encoder(latent_dim)

    decoder.build((1, latent_dim))
    encoder.build((1, 28, 28, 1))

    decoder.summary()
    encoder.summary()

    # Load data:
    (x_train, _), (x_test, _) = load_data(path='mnist.npz')

    # Normalize data:
    x_test  =  Helper.add_dim(Helper.conv_gscale(x_test, 255.0))
    x_train =  Helper.add_dim(Helper.conv_gscale(x_train, 255.0))
    mnist_digits = np.concatenate([x_test, x_train], axis=0)
    print(f"Model input dimensions: {mnist_digits.shape}")

    # Build VAE
    vae_model = VAE(encoder, decoder)

    # Setting the the flag run_eagerly to True, the models logic will not be 
    # wrapped into a tf.function.
    #
    # Problem right now: if run_eagerly is false the models logic will be wrapped 
    # into  tf.function which results to following error:
    #
    # OperatorNotAllowedInGraphError: iterating over `tf.Tensor` is not allowed: 
    # AutoGraph did convert this function. This might indicate you are trying to 
    # use an unsupported feature.
    #
    # So to solve that, run_eagerly=True was set to True. Reason for the Problem, 
    # see def train_step in class VAE()
    vae_model.compile(optimizer=tf.keras.optimizers.Adam(), run_eagerly=True)
    vae_model.fit(mnist_digits, epochs=10, batch_size=256)

    # Save the model after training
    vae_model_path = join(getcwd(), "saved_vae_model")
    tf.saved_model.save(vae_model, vae_model_path)
