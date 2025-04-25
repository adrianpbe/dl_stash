import dl_stash.rnp.rnp as rnp

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

import tensorflow_datasets as tfds


def get_ds(batch_size):
    mnist = tfds.image_classification.MNIST()
    mnist_ds = mnist.as_dataset()
    return mnist_ds["train"].map(lambda x: (tf.cast(x["image"], dtype=tf.float32) / 255.0)).batch(batch_size)


def profile_decoder(hparams: rnp.RNPHParams, batch_size: int):
    decoder = rnp.RNPDecoder(hparams)

    im_ds = get_ds(batch_size)


    iterds = iter(im_ds.take(2))

    # Take one batch of data 
    batch_images = next(iterds)
    z = tf.random.normal((batch_size, hparams.embedding_size), dtype=tf.float32)
    input_data = [batch_images, z]

    # Warmup
    output = decoder(input_data) 

    # Profiling data
    batch_images = next(iterds)
    z = tf.random.normal((batch_size, hparams.embedding_size), dtype=tf.float32)
    input_data = [batch_images, z]

    # Set up profiling
    tf.profiler.experimental.start('profiling_rnp')  # directory to save profile data

    with tf.profiler.experimental.Trace('model_call', step_num=0, _r=1):
        output = decoder(input_data)  # only this call will be profiled

    tf.profiler.experimental.stop()


def profile_train_step(hparams: rnp.RNPHParams, batch_size: int):
    
    autoencoder = rnp.RNPAutoEncoder(hparams)
    autoencoder.compile(optimizer=keras.optimizers.Adam())
    im_ds = get_ds(batch_size)


    iterds = iter(im_ds.take(2))

    # Take one batch of data 
    batch_images = next(iterds)

    input_data = batch_images
    # Warmup
    print("warming up")
    output = autoencoder.train_step(input_data) 

    # Profiling data
    batch_images = next(iterds)
    input_data = batch_images

    # Set up profiling
    print("start profiling")

    tf.profiler.experimental.start('profiling_rnp_train_step')  # directory to save profile data

    with tf.profiler.experimental.Trace('train_step', step_num=0, _r=1):
        output = autoencoder.train_step(input_data)  # only this call will be profiled

    tf.profiler.experimental.stop()


def profile_train_loop(hparams: rnp.RNPHParams, batch_size: int, num_steps: int):
    autoencoder = rnp.RNPAutoEncoder(hparams)
    autoencoder.compile(optimizer=keras.optimizers.Adam())
    im_ds = get_ds(batch_size)

    iterds = iter(im_ds.take(num_steps + 1))

    # # Take one batch of data 
    # batch_images = next(iterds)

    # input_data = batch_images
    # # Warmup
    print("warming up")
    train_data = next(iterds)
    _ = autoencoder.train_step(train_data)

    # # Profiling data
    # batch_images = next(iterds)
    # input_data = batch_images

    # Set up profiling
    print("start profiling")

    tf.profiler.experimental.start('profiling_rnp_train_loop')  # directory to save profile data

    for step in range(num_steps):
        with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
            train_data = next(iterds)
            _ = autoencoder.train_step(train_data)
        
    tf.profiler.experimental.stop()

if __name__ == "__main__":
    batch_size = 128
    hparams = rnp.RNPHParams((28, 28, 1), sequence_length=4, levels=2)
    profile_train_loop(hparams, batch_size, 4)
