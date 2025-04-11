from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from tensorflow.keras.applications import resnet
import tensorflow as tf
from tensorflow import keras
from  tensorflow.keras import layers

from dl_stash import hyper
from dl_stash.rnp.image import sample, inverse_sampling


NUM_AFFINE_TRANSFORMATION_PARAMETERS = 4


class SampleGaussian(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        mu, logvar = inputs
        noise = tf.random.normal(tf.shape(mu))
        sample = mu + tf.math.sqrt(tf.math.exp(logvar)) * noise
        return sample


def get_simple_encoder(input_shape: tuple[int, ...]) -> keras.Model:
    encoder_backbone = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(filters=8, kernel_size=3, activation="relu"),
            layers.Conv2D(filters=16, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(strides=2, padding="same"),
            layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
            layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(strides=2, padding="same"),
            layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(strides=2, padding="same"),
            layers.Conv2D(filters=256, kernel_size=3, activation="relu"),
            layers.GlobalMaxPool2D(),
            layers.Dense(256),
        ]
    )
    return encoder_backbone


def get_encoder(backbone: keras.Model, input_shape: tuple[int, ...], embedding_size: int):
    in_ = layers.Input(shape=input_shape)
    z_encoder = backbone(in_)

    z_mu = layers.Dense(embedding_size)(z_encoder)
    z_logvar = layers.Dense(embedding_size)(z_encoder)

    z_k = SampleGaussian()([z_mu, z_logvar])

    encoder = keras.Model(inputs=in_, outputs=z_k)
    return encoder


@dataclass
class RNPHParams:
    img_shape: tuple[int, int, int]
    embedding_size: int=32
    action_embedding_size: int=32
    levels: int=3
    parametrized_encoders_units: int=32
    parametrized_decoders_units: int=32
    sequence_length: int=8
    scale_offset: float=1.0 # TODO: check how it is used
    decoder_state_image_shape: tuple[int, int, int] | None = None
    hyper_decoders_units: list[int] = field(default_factory=lambda:  [64, 64])

    def __post_init__(self):
        if self.decoder_state_image_shape is None:
            self.decoder_state_image_shape = self.img_shape
        self.x_dec_size: int = int(np.prod(self.decoder_state_image_shape))


# Compute hypernetwork output size
def compute_hypernetwork_params(
        hparams: RNPHParams, decoder_size: int, transition_rnn_units: int, emb_zero_size: int):

    encoder_input_size = hparams.embedding_size + hparams.action_embedding_size
    encoder_dense1_num_params = hyper.get_total_dense_parameters(
        encoder_input_size, hparams.parametrized_encoders_units
    )
    encoder_dense2_num_params = hyper.get_total_dense_parameters(
        hparams.parametrized_encoders_units, hparams.parametrized_encoders_units
    )

    decoder_dense1_num_params = hyper.get_total_dense_parameters(
        transition_rnn_units, hparams.parametrized_decoders_units
    )
    decoder_dense2_num_params = hyper.get_total_dense_parameters(
        hparams.parametrized_decoders_units, decoder_size
    )

    # Parametrized state transition function hyper layers 
    rnn_num_params = hyper.get_total_gru_parameters(hparams.parametrized_encoders_units, transition_rnn_units)

    params = (
        encoder_dense1_num_params, encoder_dense2_num_params, 
        decoder_dense1_num_params, decoder_dense2_num_params,
        rnn_num_params, emb_zero_size
    )
    hypernetwork_output_size = int(np.sum(params))

    return hypernetwork_output_size, params


def compute_h_state_params(hparams: RNPHParams):
    return compute_hypernetwork_params(
        hparams, decoder_size=hparams.x_dec_size,
        transition_rnn_units=hparams.embedding_size, 
        emb_zero_size=hparams.embedding_size
    )


def compute_h_policy_params(hparams: RNPHParams):
    return compute_hypernetwork_params(
        hparams, decoder_size=NUM_AFFINE_TRANSFORMATION_PARAMETERS,
        transition_rnn_units=hparams.action_embedding_size, emb_zero_size=hparams.action_embedding_size
    )


def build_hypernetwork_decoder(units: list[int], embedding_size: int, output_size: int, splits: tuple[int, ...]):
    h_backbone = keras.Sequential(
        [
            layers.Input(shape=(embedding_size,), dtype=tf.float32),
        ] + [
            layers.Dense(u, activation="relu") for u in units
        ] + [
            layers.Dense(output_size)
        ]
    )

    # Model 
    z_input = layers.Input(shape=(embedding_size,), dtype=tf.float32)

    gen_params = h_backbone(z_input)

    splitted_params = layers.Lambda(
        lambda x: tf.split(x, splits, axis=1)
    )(gen_params)

    h = keras.Model(inputs=z_input, outputs=splitted_params)
    return h


def build_parametrized_models(hparams: RNPHParams, gen_params: Sequence[int],
                                    decoder_size: int, rnn_units: int):
    embedding_size = hparams.embedding_size
    action_embedding_size = hparams.action_embedding_size

    # Encoder
    enc_dense1_num_params, enc_dense2_num_params, dec_dense1_num_params, dec_dense2_num_params, rnn_num_params, _ = gen_params

    enc_dense1_params_in = layers.Input(shape=(enc_dense1_num_params,), dtype=tf.float32)
    enc_dense2_params_in = layers.Input(shape=(enc_dense2_num_params,), dtype=tf.float32)
    z_enc_input = layers.Input(shape=(embedding_size,), dtype=tf.float32)
    a_emb_enc_input = layers.Input(shape=(action_embedding_size,), dtype=tf.float32)

    x_enc = layers.Concatenate()([z_enc_input, a_emb_enc_input])
    x_enc = hyper.HyperDense(
        hparams.parametrized_encoders_units, activation="relu")([x_enc, enc_dense1_params_in])
    y_enc = hyper.HyperDense(
        hparams.parametrized_encoders_units, activation=None)([x_enc, enc_dense2_params_in])
    
    encoder = keras.Model(inputs=[z_enc_input, a_emb_enc_input, enc_dense1_params_in, enc_dense2_params_in],
                          outputs=y_enc)

    # Decoder
    dec_dense1_params_in = layers.Input(shape=(dec_dense1_num_params,), dtype=tf.float32)
    dec_dense2_params_in = layers.Input(shape=(dec_dense2_num_params,), dtype=tf.float32)
    z_dec_input = layers.Input(shape=(embedding_size,), dtype=tf.float32)

    x_dec = hyper.HyperDense(
        hparams.parametrized_decoders_units, activation="relu")([z_dec_input, dec_dense1_params_in])
    y_dec = hyper.HyperDense(
        decoder_size, activation=None)([x_dec, dec_dense2_params_in])

    decoder = keras.Model(inputs=[z_dec_input, dec_dense1_params_in, dec_dense2_params_in],
                          outputs=y_dec)

    # RNN
    rnn_input = layers.Input(shape=(hparams.parametrized_encoders_units,), dtype=tf.float32)
    rnn_params_in = layers.Input(shape=(rnn_num_params,), dtype=tf.float32)

    rnn_state_input = layers.Input(shape=(rnn_units,), dtype=tf.float32)
    gru_cell = hyper.HyperGRUCell(rnn_units,)

    y_rnn, next_state = gru_cell([rnn_input, rnn_params_in], rnn_state_input)

    #TODO: check how to deal with state nesting (it's common in Keras RNN API to have a list of states)
    rnn = keras.Model(inputs=[rnn_input, rnn_params_in, rnn_state_input], outputs=[y_rnn, next_state])

    return encoder, decoder, rnn


@dataclass
class HyperNetwork:
    hypernetwork: keras.Model
    encoder: keras.Model
    decoder: keras.Model
    rnn: keras.Model
    rnn_units: int
    states: tf.Tensor | None = None

    def reset_state(self, batch_size=None):
        if batch_size is None and self.states is None:
            raise ValueError("If states are still None batch_size is required")
        if batch_size is None:
            batch_size = tf.shape(self.state)[0]
        self.states = tf.zeros(shape=(batch_size, self.rnn_units), dtype=tf.float32)


@dataclass
class HyperNetworks:
    state: HyperNetwork
    policy: HyperNetwork


def build_hypernetworks(hparams: RNPHParams) -> HyperNetworks:
    hyper_decoders_units = hparams.hyper_decoders_units

    # State transition
    h_state_output_size, h_state_params = compute_hypernetwork_params(
        hparams, decoder_size=hparams.x_dec_size, 
        transition_rnn_units=hparams.embedding_size, emb_zero_size=hparams.embedding_size
    )
    
    h_state = build_hypernetwork_decoder(hyper_decoders_units, hparams.embedding_size,
                                          output_size=h_state_output_size,
                                          splits=h_state_params
                                          )

    parameterized_state_models = build_parametrized_models(hparams, h_state_params, 
                                                           decoder_size=hparams.x_dec_size,
                                                           rnn_units=hparams.embedding_size)
    
    state_hypernetworks = HyperNetwork(*(h_state, *parameterized_state_models, hparams.embedding_size))
    # Policy
    h_policy_output_size, h_policy_params = compute_hypernetwork_params(
        hparams, decoder_size=NUM_AFFINE_TRANSFORMATION_PARAMETERS,
        transition_rnn_units=hparams.action_embedding_size, emb_zero_size=hparams.action_embedding_size
    )

    h_policy = build_hypernetwork_decoder(hyper_decoders_units, hparams.embedding_size,
                                          output_size=h_policy_output_size,
                                          splits=h_policy_params
                                          )

    parameterized_policy_models = build_parametrized_models(hparams, h_policy_params,
                                                            decoder_size=NUM_AFFINE_TRANSFORMATION_PARAMETERS,
                                                            rnn_units=hparams.action_embedding_size)
    policy_hypernetworks = HyperNetwork(*(h_policy, *parameterized_policy_models, hparams.action_embedding_size))

    return HyperNetworks(state_hypernetworks, policy_hypernetworks)


def parametrized_hypernetwork_step(hyper: HyperNetwork, inputs, params: tuple):
    z, za = inputs
    enc_dense1_params, enc_dense2_params, dec_dense1_params, dec_dense2_params, rnn_params = params

    next_emb, hyper.states = hyper.rnn(
        [
            hyper.encoder(
                [z, za, enc_dense1_params, enc_dense2_params]
            ),
            rnn_params,
            hyper.states
        ]
    )

    dec_out = hyper.decoder(
        [next_emb, dec_dense1_params, dec_dense2_params]
    )

    return next_emb, dec_out


class RNPDecoder(keras.Model):
    def __init__(self, hparams: RNPHParams, **kwargs):
        super().__init__(**kwargs)
        self.hparams = hparams

        self.hypernetworks = build_hypernetworks(self.hparams)
        # TODO: change Hypernetwork state and policy class to a keras.Model weights for tracking!
        self.state_hyper = self.hypernetworks.state
        self.policy_hyper = self.hypernetworks.policy
    
        height, width, _ = self.hparams.img_shape
        self.sampling_grid = (height, width)

        self.reshape_out = layers.Reshape((28, 28, 1))


    def call(self, inputs):
        x, z = inputs
        batch_size = tf.shape(x)[0]
        self.state_hyper.reset_state(batch_size)
        self.policy_hyper.reset_state(batch_size)

        out = self.rnp_decoder(x, z, self.hparams.levels)
        return out

    def rnp_decoder(self, x, z, level):
        *state_generated_params, z0 = self.hypernetworks.state.hypernetwork(z)
        *policy_generated_params, za0 = self.hypernetworks.policy.hypernetwork(z)
        out = tf.zeros_like(x)
        for _ in range(self.hparams.sequence_length):
            (z0, za0), (x_hat, a), x_patch = self.step(
                x,
                inputs=(z0, za0), 
                gen_params=(state_generated_params, policy_generated_params)
            )
            if level > 0:
                out_ = self.rnp_decoder(x_patch, z, level - 1)
                out += sample(out_, a, self.sampling_grid)
            else:
                out += sample(self.reshape_out(x_hat), a, self.sampling_grid)
        return out

    def step(self, x, inputs, gen_params):
        hyper = self.hypernetworks
        state_generated_params, policy_generated_params = gen_params
        next_z, x_hat = parametrized_hypernetwork_step(hyper.state, inputs, state_generated_params)
        next_za, a = parametrized_hypernetwork_step(hyper.policy, inputs, policy_generated_params)
        # Sample the original image with the current action a, it is passed to deeper layers
        #  during recursion, useful too for generating a patch level auxiliar loss.
        x_patch = tf.stop_gradient(inverse_sampling(x, a, self.sampling_grid))
        return (next_z, next_za), (x_hat, a), x_patch
