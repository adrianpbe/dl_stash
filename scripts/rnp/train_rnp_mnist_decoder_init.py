import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tqdm import tqdm

from dl_stash.rnp.rnp import RNPHParams, RNPDecoder, RNPAutoEncoder

import os

# tf.config.set_visible_devices([], 'GPU')


def preprocess_data(x):
    """Preprocess MNIST images."""
    # Convert to float32 and normalize to [0, 1]
    image = tf.cast(x["image"], dtype=tf.float32) / 255.0
    return image


def get_decoder_ds_and_length():
    mnist_ds = tfds.image_classification.MNIST().as_dataset()["train"].map(preprocess_data)
    ds_length = sum(1 for _ in mnist_ds)
    decoder_ds = tf.data.Dataset.zip((mnist_ds, tf.data.Dataset.from_tensor_slices(np.arange(ds_length))))
    return decoder_ds, ds_length



def save_reconstruction_samples(model, data_batch, epoch, save_dir):
    """Save reconstructed images."""
    # Randomly select indices
    
    images, original_indices = data_batch
    if len(images) > 16:
        raise ValueError("the batch length must smaller or equal than 16")
    
    reconstructions = model(data_batch)
    original_indices = original_indices.numpy()
    images = images.numpy()
    
    
    # Create figure
    plt.figure(figsize=(20, 10))
    for i, idx in enumerate(original_indices):
        # While filling a row of Original images, a second row of Recons
        #  is also being filled, so for the next row a "jump" is needed,
        #  this is achieved by adding the offset
        offset = i // 4 * 4
        plt.subplot(8, 4, offset + i + 1)
        plt.imshow(images[i, :, :, 0], cmap='gray')
        plt.title(f"Original {str(idx)}")
        plt.axis('off')
        
        plt.subplot(8, 4, offset + i + 1 + 4)
        plt.imshow(reconstructions[i, :, :, 0], cmap='gray')
        plt.title(f"Recons {str(idx)}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'reconstruction_epoch_{epoch}.png'))
    plt.close()


class SaveReconstructionCallback(keras.callbacks.Callback):
    """Callback to save reconstruction samples during training."""
    def __init__(self, batch, save_dir):
        super().__init__()
        self.batch = batch
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs=None):
        save_reconstruction_samples(
            self.model,
            self.batch,
            epoch,
            self.save_dir,
        )

class MultiDecoderTrainer(keras.Model):
    def __init__(self, hparams: RNPHParams, num_images, **kwargs):
        super().__init__(**kwargs)
        self.hparams = hparams
        self.decoder = RNPDecoder(hparams)
        # Create a learnable embedding for each image
        self.image_embeddings = tf.Variable(
            tf.random.normal((num_images, hparams.embedding_size), dtype=tf.float32),
            trainable=True, name="embs"
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
    
    def call(self, x):
        # x is a tuple of (images, indices)
        images, indices = x
        # Get the corresponding embedding for each index
        z_batch = tf.gather(self.image_embeddings, indices)
        return self.decoder([images, z_batch])

    @tf.function
    def train_step(self, data):
        images, idxs = data
        with tf.GradientTape() as tape:
            reconstruction = self.call(data)
            loss = tf.math.reduce_mean(
                tf.math.reduce_sum(
                    tf.math.square(images - reconstruction),
                     axis=(1,2,3)
                ),
            )
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        reconstruction = self(data)
        reconstruction_loss = tf.math.reduce_mean(
                tf.math.reduce_sum(
                    tf.math.square(data - reconstruction),
                     axis=(1,2,3)
                ),
            )
        self.loss_tracker.update_state(reconstruction_loss)
        return {
            "loss": self.loss_tracker.result(),
        }


def extract_reconstruction_monitoring_batch(ds):
    ims, idx = next(iter(ds))
    return ims[:16], idx[:16]


def main():
    # Set hyperparameters
    BATCH_SIZE = 256
    decoder_epochs = 20
    decoder_lr = 1e-3
    epochs = 20
    lr = 1e-4
    FREEZE_DECODER = True

    IM_PATH = "data/initialized/1"
    os.makedirs(IM_PATH, exist_ok=True)

    hparams = RNPHParams(
        img_shape=(28, 28, 1),  # MNIST image shape
        embedding_size=64,
        action_embedding_size=64,
        levels=2,  # Number of recursive levels
        sequence_length=3,  # Number of steps in the sequence
        parametrized_encoders_units=64,
        parametrized_decoders_units=64,
        hyper_decoders_units=[64, 64, 64, 64],
        scale_offset=0,
        lambda_=5e-4,
    )
    filter_fn = lambda x: (x["label"] == 0 or x["label"] == 1)
    print("Training decoder")
    decoder_ds, len_ds = get_decoder_ds_and_length()
    decoder_ds = decoder_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    decoder = MultiDecoderTrainer(hparams, num_images=len_ds)
    decoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=decoder_lr),
        loss='mse',
        run_eagerly=True,
    )

    decoder.fit(decoder_ds, epochs=decoder_epochs, 
                callbacks=[SaveReconstructionCallback(
                    extract_reconstruction_monitoring_batch(decoder_ds),
                    IM_PATH
                    )]
                )

    # Autoencoder
    mnist = tfds.image_classification.MNIST()
    mnist_ds = mnist.as_dataset()

    train_ds = (
        mnist_ds["train"]
        .filter(filter_fn)
        .map(preprocess_data)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        mnist_ds["test"]
        .filter(filter_fn)
        .map(preprocess_data)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    model = RNPAutoEncoder(hparams)
    model.build((None,) + hparams.img_shape)
    model.decoder.set_weights(decoder.decoder.get_weights())

    assert all(
        np.array_equal(a, b) for a, b in zip(
            model.decoder.get_weights(), 
            decoder.decoder.get_weights()
            )
    ), "weights not properly copied from pretrained decoder!"

    model.decoder.trainable = not FREEZE_DECODER

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        run_eagerly=True
    )

    history = model.fit(train_ds, epochs=epochs, validation_data=test_ds)

    # Save the model
    model.save_weights("pretrained_rnp_mnist_zero_or_one.weights.h5")

    # Print final metrics
    print("\nTraining completed!")
    if not isinstance(history, dict):
        history= history.history
    print(f"Final training loss: {history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
