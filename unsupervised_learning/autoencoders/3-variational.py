#!/usr/bin/env python3
"""Variational autoencoder"""
import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    input_dims: integer, dimensions of the model input
    hidden_layers: list, number of nodes for each hidden layer in the
                   encoder (reversed for the decoder)
    latent_dims: integer, dimensions of the latent space representation

    Returns: encoder, decoder, auto
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        mean, log_var = args
        epsilon = K.random_normal(shape=K.shape(mean))
        return mean + K.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z, z_mean, z_log_var])

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs)

    # Full autoencoder
    auto_inputs = keras.Input(shape=(input_dims,))
    z, z_mean, z_log_var = encoder(auto_inputs)
    decoder_outputs = decoder(z)
    auto = keras.Model(auto_inputs, decoder_outputs)

    def vae_loss(inputs, outputs):
        reconstruction_loss = keras.losses.binary_crossentropy(
            inputs, outputs)
        reconstruction_loss *= input_dims
        kl_loss = -0.5 * K.sum(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(reconstruction_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
