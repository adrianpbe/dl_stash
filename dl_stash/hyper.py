"""Implements hypernetwork versions of popular layers"""
import numpy as np


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.util import nest
from tensorflow.python.keras import activations


def num_params_from_shapes(param_shapes):
    total_params = np.sum(
        [np.prod(s) 
         for s in param_shapes
         if s is not None]
    )
    
    return total_params


def unstack_gru_params(parameters, parameters_shapes):
    batch_size = tf.shape(parameters)[0]
    kernel_shape, recurrent_kernel_shape, bias_shape = parameters_shapes

    num_kernel_params = tf.math.reduce_prod(kernel_shape)
    num_r_kernel_params = tf.math.reduce_prod(recurrent_kernel_shape)

    kernel = parameters[:, :num_kernel_params]
    kernel = tf.reshape(kernel, [batch_size, *kernel_shape])

    offset = num_kernel_params
    recurrent_kernel = parameters[:, offset:offset + num_r_kernel_params]
    recurrent_kernel = tf.reshape(recurrent_kernel, [batch_size, *recurrent_kernel_shape])

    if bias_shape is not None:
        offset += num_r_kernel_params
        num_bias = tf.math.reduce_prod(bias_shape)
        bias = parameters[:, offset:offset+num_bias]
        bias = tf.reshape(bias, [batch_size, *bias_shape])
    else:
        bias = None
    return kernel, recurrent_kernel, bias


def get_gru_parameters_shapes(input_shape: int | tuple | tf.TensorShape, units, use_bias=True, reset_after=True):
    if not isinstance(input_shape, int):
        input_dim = input_shape[-1]
    else:
        input_dim = input_shape

    kernel_shape = (input_dim, units * 3)
    recurrent_kernel_shape = (units, units * 3)

    if use_bias:
        if not reset_after:
            bias_shape = (3 * units,)
        else:
            # separate biases for input and recurrent kernels
            # Note: the shape is intentionally different from CuDNNGRU biases
            # `(2 * 3 * self.units,)`, so that we can distinguish the classes
            # when loading and converting saved weights.
            bias_shape = (2, 3 * units)
    else:
        bias_shape = None
    
    return [kernel_shape, recurrent_kernel_shape, bias_shape]


def get_total_gru_parameters(input_size, rnn_units, use_bias=True, reset_after=True):
    param_shapes = get_gru_parameters_shapes(input_size, rnn_units, use_bias, reset_after)
    total_params = num_params_from_shapes(param_shapes)
    return total_params



class HyperGRUCell(layers.Layer):
    """A Hypernetwork version of GRUCell, call takes as inputs the regular 
    GRU inputs as well as a Tensor with the parameters, so the layer does not
    have any trainable parameter."""
    def __init__(self,
                 units,
                 activation="tanh",
                 recurrent_activation="hard_sigmoid",
                 use_bias=True,
                 reset_after=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.reset_after = reset_after
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        implementation = kwargs.pop("implementation", 2)
        self.recurrent_dropout = 0
        if self.recurrent_dropout != 0 and implementation != 1:
            self.implementation = 1
        else:
            self.implementation = implementation
      
        self.state_size = self.units
        self.output_size = self.units

    def build(self, input_shape):
        # When building the cell directly input_shape is list[TensorShape], while when building
        #  from RNN it is a tuple with the shape of the first input (param import is ignored) I think I"m missing
        #  something from RNN build implementation or it is simply bugged!
        # So I"m solving it with this check, beware if using a different TF version!
        if isinstance(input_shape, (tuple, list)):
            if isinstance(input_shape[0], (tuple, list, tf.TensorShape)):
                input_shape = input_shape[0]

        self.parameters_shapes = get_gru_parameters_shapes(
            input_shape, self.units, use_bias=self.use_bias, reset_after=self.reset_after
        )
        self.built = True

    def call(self, inputs, states, training=None):
        inputs, parameters = inputs
        h_tm1 = states[0] if nest.is_nested(states) else states  # previous memory
        kernel, recurrent_kernel, bias = unstack_gru_params(parameters, self.parameters_shapes)

        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = bias, None
            else:
                input_bias, recurrent_bias = tf.unstack(bias, axis=1,)

        if self.implementation == 1:
            inputs_z = inputs
            inputs_r = inputs
            inputs_h = inputs
 
            x_z = tf.einsum("...ij,...ijk->...ik", inputs_z, kernel[..., :self.units])
            x_r = tf.einsum("...ij,...ijk->...ik", inputs_r, kernel[..., self.units:self.units * 2])
            x_h = tf.einsum("...ij,...ijk->...ik", inputs_h, kernel[..., self.units * 2:])

            if self.use_bias:
                x_z = tf.math.add(x_z, input_bias[..., :self.units])
                x_r = tf.math.add(x_r, input_bias[..., self.units: self.units * 2])
                x_h = tf.math.add(x_h, input_bias[..., self.units * 2:])

            h_tm1_z = h_tm1
            h_tm1_r = h_tm1
            h_tm1_h = h_tm1

            recurrent_z = tf.einsum("...ij,...ijk->...ik", h_tm1_z, recurrent_kernel[..., :self.units])
            recurrent_r = tf.einsum("...ij,...ijk->...ik", 
                    h_tm1_r, recurrent_kernel[..., self.units:self.units * 2])
            if self.reset_after and self.use_bias:
                recurrent_z = tf.math.add(recurrent_z, recurrent_bias[..., :self.units])
                recurrent_r = tf.math.add(
                        recurrent_r, recurrent_bias[..., self.units:self.units * 2])

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = tf.einsum("...ij,...ijk->...ik", 
                        h_tm1_h, recurrent_kernel[..., self.units * 2:])
                if self.use_bias:
                    recurrent_h = tf.math.add(
                            recurrent_h, recurrent_bias[..., self.units * 2:])
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = tf.einsum("...ij,...ijk->...ik", 
                        r * h_tm1_h, recurrent_kernel[..., self.units * 2:])

            hh = self.activation(x_h + recurrent_h)
        else:
            # if 0. < self.dropout < 1.:
            #     inputs = inputs * dp_mask[0]

            # inputs projected by all gate matrices at once
            matrix_x = tf.einsum("...ij,...ijk->...ik", inputs, kernel)
            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                matrix_x = tf.math.add(matrix_x, input_bias)

            x_z, x_r, x_h = tf.split(matrix_x, 3, axis=-1)

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner = tf.einsum("...ij,...ijk->...ik", h_tm1, recurrent_kernel)
                if self.use_bias:
                    matrix_inner = tf.math.add(matrix_inner, recurrent_bias)
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner = tf.einsum("...ij,...ijk->...ik", 
                        h_tm1, recurrent_kernel[..., :2 * self.units])

            recurrent_z, recurrent_r, recurrent_h = tf.split(
                    matrix_inner, [self.units, self.units, -1], axis=-1)

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            if self.reset_after:
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = tf.einsum("...ij,...ijk->...ik", 
                        r * h_tm1, recurrent_kernel[..., 2 * self.units:])

            hh = self.activation(x_h + recurrent_h)
        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh
        new_state = [h] if nest.is_nested(states) else h
        return h, new_state

    def get_config(self):
        config = super(HyperGRUCell, self).get_config()
        config.update({
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation),
            "recurrent_activation": tf.keras.activations.serialize(self.recurrent_activation),
            "use_bias": self.use_bias,
            "reset_after": self.reset_after,
        })
        return config

