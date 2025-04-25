import os

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from dl_stash.rnp.rnp import RNPHParams, RNPDecoder


def get_single_image_per_class():
    """Get one MNIST image for each class (0-9)."""
    mnist = tfds.image_classification.MNIST()
    mnist_ds = mnist.as_dataset()
    
    # Get one image per class
    images = {}
    labels = {}
    for example in mnist_ds["test"]:
        label = example["label"].numpy()
        if label not in images:
            images[label] = tf.cast(example["image"], dtype=tf.float32) / 255.0
            labels[label] = label
            if len(images) == 10:  # We have one image for each class
                break
    
    # Stack images and labels
    images = tf.stack([images[i] for i in range(10)])
    labels = tf.stack([labels[i] for i in range(10)])
    return images, labels


class ClassDecoderTrainer(keras.Model):
    def __init__(self, hparams: RNPHParams, num_classes: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.hparams = hparams
        self.decoder = RNPDecoder(hparams)
        # Create a learnable embedding for each class
        self.class_embeddings = tf.Variable(
            tf.random.normal((num_classes, hparams.embedding_size), dtype=tf.float32),
            trainable=True
        )
        
    def call(self, x):
        # x is a tuple of (images, labels)
        images, labels = x
        # Get the corresponding embedding for each label
        z_batch = tf.gather(self.class_embeddings, labels)
        return self.decoder([images, z_batch])


def main():
    IM_PATH = "rnp_cls_1"
    os.makedirs(IM_PATH, exist_ok=True)
    
    # Set hyperparameters
    hparams = RNPHParams(
        img_shape=(28, 28, 1),  # MNIST image shape
        embedding_size=16,  # Increased embedding size to capture class information
        action_embedding_size=16,
        levels=2,
        sequence_length=3,
        parametrized_encoders_units=64,
        parametrized_decoders_units=64,
        hyper_decoders_units=[64, 64]
    )

    # Get one image per class
    target_images, target_labels = get_single_image_per_class()
    
    # Create model
    model = ClassDecoderTrainer(hparams)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse'
    )
    model.build((None,) + hparams.img_shape)
    
    # Training loop
    print("Starting training...")
    for epoch in range(10000):
        with tf.GradientTape() as tape:
            reconstruction = model((target_images, target_labels))
            loss = tf.reduce_mean(tf.square(target_images - reconstruction))
        
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")
            
            # Visualize reconstructions for each class
            plt.figure(figsize=(20, 10))
            for i in range(10):
                plt.subplot(2, 10, i + 1)
                plt.imshow(target_images[i, :, :, 0], cmap='gray')
                plt.title(f'Original {i}')
                plt.axis('off')
                
                plt.subplot(2, 10, i + 11)
                plt.imshow(reconstruction[i, :, :, 0], cmap='gray')
                plt.title(f'Recon {i}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(IM_PATH, f'reconstruction_epoch_{epoch}.png')
            )
            plt.close()


if __name__ == "__main__":
    main()
