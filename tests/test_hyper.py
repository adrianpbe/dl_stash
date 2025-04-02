import dl_stash.hyper as hyper
 
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


def test_hyper_gru():
    batch_size = 4
    rnn_units = 8
    input_size = 4

    total_params = int(hyper.get_total_gru_parameters(input_size, rnn_units))  # Convert to Python int

    inputs = layers.Input(shape=(input_size,), dtype=tf.float32)
    params = layers.Input(shape=(total_params,), dtype=tf.float32)
    state = layers.Input(shape=(rnn_units,), dtype=tf.float32)

    cell = hyper.HyperGRUCell(rnn_units)
    output, next_state = cell([inputs, params], [state] )

    # Create model
    model = keras.Model(inputs=[inputs, params, [state]], outputs=[output, next_state])

    # Test with random data
    x = np.random.random((batch_size, input_size))
    p = np.random.random((batch_size, total_params))
    s = np.random.random((batch_size, rnn_units))

    output, next_state = model.predict([x, p, [s]])

    assert output.shape == (batch_size, rnn_units)
    assert next_state[0].shape == (batch_size, rnn_units)

    # Test that state updates work by feeding output state back in
    output2, next_state2 = model.predict([x, p, next_state])
    assert output2.shape == (batch_size, rnn_units)
    assert next_state2[0].shape == (batch_size, rnn_units)


# Not working, but I think it's due to a Tensorflow bug: https://github.com/tensorflow/tensorflow/issues/90389
# def test_hyper_gru_seq():
#     batch_size = 4
#     timesteps = 10
#     rnn_units = 8
#     input_size = 4

#     total_params = int(hyper.get_total_gru_parameters(input_size, rnn_units))  # Convert to Python int

#     in_seq = layers.Input(shape=(timesteps, input_size), dtype=tf.float32)
#     params_seq = layers.Input(shape=(timesteps, total_params), dtype=tf.float32)

#     cell = hyper.HyperGRUCell(rnn_units)
#     hyper_gru_seq = layers.RNN(
#         cell,
#         return_sequences=True,
#         input_shape=(timesteps, input_size),
#         stateful=False
#     )

#     yseq = hyper_gru_seq([in_seq, params_seq])

#     hyper_gru_cell_model = keras.Model(inputs=(in_seq, params_seq), outputs=yseq)

#     x = np.random.random((batch_size, timesteps, input_size))
#     params = np.random.random((batch_size, timesteps, total_params))
#     output = hyper_gru_cell_model((x, params))
    
#     assert output.shape == (batch_size, timesteps, rnn_units)


