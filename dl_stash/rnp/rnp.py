from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from tensorflow.keras.applications import resnet
import tensorflow as tf
from tensorflow import keras
from  tensorflow.keras import layers

from dl_stash import hyper
import dl_stash.rnp.image



NUM_AFFINE_TRANSFORMATION_PARAMETERS = 6


def sample_patch(image, params, sampling_grid, scale_offset: float):
    
    default_scale_offset = tf.convert_to_tensor([[0, 0, 0, 0, 1, 1]], dtype=tf.float32)
    offset = default_scale_offset + scale_offset * tf.convert_to_tensor([[0, 0, 0, 0, 1, 1]], dtype=tf.float32)

    params = params + offset

    return dl_stash.rnp.image.sample(image, params, sampling_grid)


def inverse_sample_patch(image, params, sampling_grid, scale_offset: float):

    default_scale_offset = tf.convert_to_tensor([[0, 0, 0, 0, 1, 1]], dtype=tf.float32)
    offset = scale_offset * default_scale_offset
    
    params = params + offset

    return dl_stash.rnp.image.inverse_sampling(image, params, sampling_grid)


class SampleGaussian(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        mu, logvar = inputs
        noise = tf.random.normal(tf.shape(mu))
        sample = mu + tf.math.sqrt(tf.math.exp(logvar)) * noise
        return sample


def get_simple_encoder(input_shape: tuple[int, int, int]) -> keras.Model:
    encoder_backbone = keras.Sequential(
        [
            layers.Input(shape=input_shape, dtype=tf.float32),
            layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
            layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(strides=2, padding="same"),
            layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
            layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(strides=2, padding="same"),
            layers.GlobalMaxPool2D(),
            layers.Dense(128),
        ]
    )
    return encoder_backbone


def get_encoder(backbone: keras.Model, input_shape: tuple[int, ...], embedding_size: int, **kwargs):
    in_ = layers.Input(shape=input_shape)
    z_encoder = backbone(in_)

    z_mu = layers.Dense(embedding_size)(z_encoder)
    z_logvar = layers.Dense(embedding_size)(z_encoder)

    z_k = SampleGaussian()([z_mu, z_logvar])

    encoder = keras.Model(inputs=in_, outputs=[z_k, z_mu, z_logvar], **kwargs)
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
    scale_offset: float=1.6
    beta: float=0.1
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


def build_hypernetwork_decoder(units: list[int], embedding_size: int, output_size: int, splits: tuple[int, ...], name):
    h_backbone = keras.Sequential(
        [
            layers.Input(shape=(embedding_size,), dtype=tf.float32),
        ] + [
            layers.Dense(u, activation="relu") for u in units
        ] + [
            layers.Dense(output_size)
        ],
        name=name + "_backbone"
    )

    # Model 
    z_input = layers.Input(shape=(embedding_size,), dtype=tf.float32)

    gen_params = h_backbone(z_input)

    splitted_params = layers.Lambda(
        lambda x: tf.split(x, splits, axis=1)
    )(gen_params)

    h = keras.Model(inputs=z_input, outputs=splitted_params, name=name)
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
                          outputs=y_enc, name="encoder")

    # Decoder
    dec_dense1_params_in = layers.Input(shape=(dec_dense1_num_params,), dtype=tf.float32)
    dec_dense2_params_in = layers.Input(shape=(dec_dense2_num_params,), dtype=tf.float32)
    z_dec_input = layers.Input(shape=(embedding_size,), dtype=tf.float32)

    x_dec = hyper.HyperDense(
        hparams.parametrized_decoders_units, activation="relu")([z_dec_input, dec_dense1_params_in])
    y_dec = hyper.HyperDense(
        decoder_size, activation=None)([x_dec, dec_dense2_params_in])

    decoder = keras.Model(inputs=[z_dec_input, dec_dense1_params_in, dec_dense2_params_in],
                          outputs=y_dec, name="decoder")

    # RNN
    rnn_input = layers.Input(shape=(hparams.parametrized_encoders_units,), dtype=tf.float32)
    rnn_params_in = layers.Input(shape=(rnn_num_params,), dtype=tf.float32)

    rnn_state_input = layers.Input(shape=(rnn_units,), dtype=tf.float32)
    gru_cell = hyper.HyperGRUCell(rnn_units,)

    y_rnn, next_state = gru_cell([rnn_input, rnn_params_in], rnn_state_input)

    #TODO: check how to deal with state nesting (it's common in Keras RNN API to have a list of states)
    rnn = keras.Model(inputs=[rnn_input, rnn_params_in, rnn_state_input], outputs=[y_rnn, next_state], name="rnn")

    return encoder, decoder, rnn


class HyperNetwork(keras.Model):
    def __init__(self, hypernetwork: keras.Model, encoder: keras.Model, 
                 decoder: keras.Model, rnn: keras.Model, rnn_units: int, **kwargs):
        super().__init__(**kwargs)
        self.hypernetwork = hypernetwork
        self.encoder = encoder
        self.decoder = decoder
        self.rnn = rnn
        self.rnn_units = rnn_units
        self.built = True

    @tf.function
    def parametrized_hypernetwork_step(self, inputs, params, states):
        z, za = inputs
        enc_dense1_params, enc_dense2_params, dec_dense1_params, dec_dense2_params, rnn_params = params

        encoder_output = self.encoder([z, za, enc_dense1_params, enc_dense2_params])
        next_emb, next_states = self.rnn([encoder_output, rnn_params, states])
        dec_out = self.decoder([next_emb, dec_dense1_params, dec_dense2_params])

        return next_emb, dec_out, next_states

    def call(self, z):
        return self.hypernetwork(z)


@dataclass
class HyperNetworks:
    state: HyperNetwork
    policy: HyperNetwork


def build_hypernetworks(hparams: RNPHParams) -> tuple[HyperNetwork, HyperNetwork]:
    hyper_decoders_units = hparams.hyper_decoders_units

    # State transition
    h_state_output_size, h_state_params = compute_hypernetwork_params(
        hparams, decoder_size=hparams.x_dec_size, 
        transition_rnn_units=hparams.embedding_size, emb_zero_size=hparams.embedding_size
    )
    
    h_state = build_hypernetwork_decoder(hyper_decoders_units, hparams.embedding_size,
                                          output_size=h_state_output_size,
                                          splits=h_state_params,
                                          name="H_state"
                                          )

    parameterized_state_models = build_parametrized_models(hparams, h_state_params, 
                                                           decoder_size=hparams.x_dec_size,
                                                           rnn_units=hparams.embedding_size)
    
    state_hypernetworks = HyperNetwork(*(h_state, *parameterized_state_models, hparams.embedding_size), 
                                       name="state_hypernetwork"
                                       )
    # Policy
    h_policy_output_size, h_policy_params = compute_hypernetwork_params(
        hparams, decoder_size=NUM_AFFINE_TRANSFORMATION_PARAMETERS,
        transition_rnn_units=hparams.action_embedding_size, emb_zero_size=hparams.action_embedding_size
    )

    h_policy = build_hypernetwork_decoder(hyper_decoders_units, hparams.embedding_size,
                                          output_size=h_policy_output_size,
                                          splits=h_policy_params,
                                          name="H_policy"
                                          )

    parameterized_policy_models = build_parametrized_models(hparams, h_policy_params,
                                                            decoder_size=NUM_AFFINE_TRANSFORMATION_PARAMETERS,
                                                            rnn_units=hparams.action_embedding_size)
    policy_hypernetworks = HyperNetwork(*(h_policy, *parameterized_policy_models, hparams.action_embedding_size),
                                        name="policy_hypernetwork")

    return state_hypernetworks, policy_hypernetworks


class RNPDecoder(keras.Model):
    def __init__(self, hparams: RNPHParams, **kwargs):
        super().__init__(**kwargs)
        self.hparams = hparams

        self.state_hyper, self.policy_hyper = build_hypernetworks(self.hparams)
    
        height, width, _ = self.hparams.img_shape
        self.sampling_grid = (height, width)
        self.reshape_out = layers.Reshape(self.hparams.decoder_state_image_shape)
        if self.hparams.zk_shortcut:
            self.shortcut_add = layers.Add()
    
    def compute_output_shape(self, input_shape):
        # Input is [x, z] where x has the same shape as output
        return input_shape[0]
    
    def call(self, inputs):
        x, z = inputs
        batch_size = tf.shape(x)[0]
        out = self.rnp_decoder(x, z, self.hparams.levels, batch_size)
        return out

    @tf.function
    def rnp_decoder(self, x, z, level, batch_size):
        # Remove @tf.function decorator to avoid issues with variable creation
        *state_generated_params, z0 = self.state_hyper(z)
        *policy_generated_params, za0 = self.policy_hyper(z)
        rnn_states = self.initial_rnn_steps(batch_size)
        out = tf.zeros_like(x)
        for _ in range(self.hparams.sequence_length):
            (z0, za0), (x_hat, a), x_patch, rnn_states = self.step(
                x,
                inputs=(z0, za0), 
                gen_params=(state_generated_params, policy_generated_params),
                rnn_states=rnn_states
            )
            if level > 0:
                if self.hparams.zk_shortcut:
                    z0rec = self.shortcut_add([z, z0])
                else:
                    z0rec = z0
                out_ = self.rnp_decoder(x_patch, z0rec, level - 1, batch_size)
                out += sample_patch(out_, a, self.sampling_grid, self.hparams.scale_offset)
            else:
                out += sample_patch(self.reshape_out(x_hat), a, self.sampling_grid, self.hparams.scale_offset)
        return out

    def initial_rnn_steps(self, batch_size):
        return (
            tf.zeros(shape=(batch_size, self.state_hyper.rnn_units), dtype=tf.float32),
            tf.zeros(shape=(batch_size, self.policy_hyper.rnn_units), dtype=tf.float32)
        )

    @tf.function
    def step(self, x, inputs, gen_params, rnn_states):
        state_rnn_states, policy_rnn_states = rnn_states
        state_generated_params, policy_generated_params = gen_params
        next_z, x_hat, state_rnn_states = self.state_hyper.parametrized_hypernetwork_step(
            inputs, state_generated_params, state_rnn_states
        )
        next_za, a, policy_rnn_states = self.policy_hyper.parametrized_hypernetwork_step(
            inputs, policy_generated_params, policy_rnn_states
        )    
        # Sample the original image with the current action a, it is passed to deeper layers
        #  during recursion, useful too for generating a patch level auxiliar loss.
        x_patch = tf.stop_gradient(inverse_sample_patch(x, a, self.sampling_grid, self.hparams.scale_offset))
        return (next_z, next_za), (x_hat, a), x_patch, (state_rnn_states, policy_rnn_states)


class RNPAutoEncoder(keras.Model):
    def __init__(self, hparams: RNPHParams, **kwargs):
        super().__init__(**kwargs)
        self.hparams = hparams
        self.encoder = get_encoder(
            get_simple_encoder(hparams.img_shape),
            hparams.img_shape,
            hparams.embedding_size,
            name="encoder"
        )
        self.decoder = RNPDecoder(hparams, name="rnp_decoder")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
        # # Programmatically wrap the train step, this is done
        # #  to wrap the `train_step` method in a way that tf.function
        # #  can take into account the input shape of the model (from self.hparams).
        # #  This would be impossible if simply using a @tf.function decorator!
        # self.train_step = tf.function(
        #     self.train_step,
        #     input_signature=[
        #         tf.TensorSpec(shape=(None,) + self.hparams.img_shape)
        #     ],
        #     reduce_retracing=True,
        # )

    def  build(self, input_shape):
        super().build(input_shape)
        self.built = True

    @tf.function(reduce_retracing=True)
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_k, z_mu, z_logvar = self.encoder(data)
            reconstruction = self.decoder([data, z_k])
            reconstruction_loss = tf.math.reduce_mean(
                tf.math.reduce_sum(
                    tf.math.square(data - reconstruction),
                    axis=(1, 2, 3)
                )
            )
            kl_loss = -0.5 * (1 + z_logvar - tf.math.square(z_mu) - tf.math.exp(z_logvar))
            kl_loss = tf.math.reduce_sum(tf.math.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.hparams.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @tf.function(reduce_retracing=True)
    def test_step(self, data):
        z_k, z_mu, z_logvar = self.encoder(data)
        reconstruction = self.decoder([data, z_k])
        reconstruction_loss = tf.math.reduce_mean(
            tf.math.reduce_sum(
                tf.math.square(data - reconstruction),
                axis=(1, 2, 3)
            )
        )
        kl_loss = -0.5 * (1 + z_logvar - tf.math.square(z_mu) - tf.math.exp(z_logvar))
        kl_loss = tf.math.reduce_sum(tf.math.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, x):
        z_k, z_mu, z_logvar = self.encoder(x)
        y = self.decoder([x, z_k])
        return y
