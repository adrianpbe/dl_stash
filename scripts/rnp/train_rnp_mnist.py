import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tqdm import tqdm

from dl_stash.rnp.rnp import RNPHParams, RNPAutoEncoder


# tf.config.set_visible_devices([], 'GPU')


def preprocess_data(x):
    """Preprocess MNIST images."""
    # Convert to float32 and normalize to [0, 1]
    image = tf.cast(x["image"], dtype=tf.float32) / 255.0
    return image


def main():
    # Set hyperparameters
    # BATCH_SIZE = 32
    BATCH_SIZE = 256
    epochs = 50

    hparams = RNPHParams(
        img_shape=(28, 28, 1),  # MNIST image shape
        embedding_size=16,
        action_embedding_size=16,
        levels=2,  # Number of recursive levels
        sequence_length=3,  # Number of steps in the sequence
        parametrized_encoders_units=64,
        parametrized_decoders_units=64,
        hyper_decoders_units=[64, 64, 64],
        scale_offset=0
    )

    mnist = tfds.image_classification.MNIST()
    mnist_ds = mnist.as_dataset()

    train_ds = (
        mnist_ds["train"]
        .filter(lambda x: (x["label"] == 0 or x["label"] == 1))
        .map(preprocess_data)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        mnist_ds["test"]
        .filter(lambda x: (x["label"] == 0 or x["label"] == 1))
        .map(preprocess_data)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = RNPAutoEncoder(hparams)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        run_eagerly=True
    )
    # Have to run this build or weights won't be saved!
    model.build((None,) + hparams.img_shape)
    # Define callbacks
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir="./logs/rnp",
            histogram_freq=1
        )
    ]

    history = model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=test_ds)

    # Save the model
    model.save_weights("rnp_mnist_zero_or_one.weights.h5")

    # Print final metrics
    print("\nTraining completed!")
    if not isinstance(history, dict):
        history= history.history
    print(f"Final training loss: {history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
