import os

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

from dl_stash.rnp.rnp import RNPHParams, RNPDecoder


def get_single_image():
    """Get a single MNIST image for testing."""
    mnist = tfds.image_classification.MNIST()
    mnist_ds = mnist.as_dataset()
    image = next(iter(mnist_ds["test"]))["image"]
    image = tf.cast(image, dtype=tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image


class DecoderTrainer(keras.Model):
    def __init__(self, hparams: RNPHParams, **kwargs):
        super().__init__(**kwargs)
        self.hparams = hparams
        self.decoder = RNPDecoder(hparams)
        self.z = self.add_weight(
            shape=(1, hparams.embedding_size), dtype=tf.float32,
            initializer="random_normal",
            trainable=True,
            name="embedding"
        )
        
    def call(self, x):
        # Use the same learnable code for all inputs
        z_batch = tf.tile(self.z, [tf.shape(x)[0], 1])
        return self.decoder([x, z_batch])


def main():
    IM_PATH = "r_10k"
    os.mkdir(IM_PATH)
    # Set hyperparameters
    hparams = RNPHParams(
        img_shape=(28, 28, 1),  # MNIST image shape
        embedding_size=8,
        action_embedding_size=8,
        levels=2,  # Keep it simple for testing
        sequence_length=2,
        parametrized_encoders_units=32,
        parametrized_decoders_units=32,
        hyper_decoders_units=[64, 64]
    )

    # Get single image
    target_image = get_single_image()
    
    # Create model
    model = DecoderTrainer(hparams)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse'
    )
    model.build((None,) + hparams.img_shape)
    
    # Training loop
    print("Starting training...")
    for epoch in range(10000):
        with tf.GradientTape() as tape:
            reconstruction = model(target_image)
            loss = tf.reduce_mean(tf.square(target_image - reconstruction))
        
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")
            
            # Visualize reconstruction
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(target_image[0, :, :, 0], cmap='gray')
            plt.title('Original')
            plt.subplot(1, 2, 2)
            plt.imshow(reconstruction[0, :, :, 0], cmap='gray')
            plt.title('Reconstruction')
            plt.savefig(
                os.path.join(IM_PATH, f'reconstruction_epoch_{epoch}.png')
            )
            plt.close()


if __name__ == "__main__":
    main()
