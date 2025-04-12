import dl_stash.rnp.rnp as rnp

import tensorflow as tf


def test_rnp_decoder_different_batch_size():
                    
    hparams = rnp.RNPHParams((28, 28, 1), sequence_length=2, levels=2)
    decoder = rnp.RNPDecoder(hparams)

    batch_size = 1
    # Generate random image data
    batch_images = tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)
    # Generate random embedding
    z = tf.random.normal((batch_size, hparams.embedding_size), dtype=tf.float32)
    input_data = [batch_images, z]
    output = decoder(input_data) 

    batch_size = 4
    # Generate random image data with new batch size
    batch_images = tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)
    # Generate random embedding
    z = tf.random.normal((batch_size, hparams.embedding_size), dtype=tf.float32)
    input_data = [batch_images, z]
    output = decoder(input_data)

    assert True


def test_get_rnp_autoencoder():
    hparams = rnp.RNPHParams((28, 28, 1), sequence_length=4, levels=2)
    model =  rnp.get_rnp_autoencoder(hparams)
    assert True

