from tensorflow import keras
from tensorflow.keras import layers


class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, shortcut_conv, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.shortcut_conv = shortcut_conv
        self.activation = keras.activations.get(activation)
        self.conv1 = layers.Conv2D(filters, kernel_size, padding="same",
                                   name="conv1", strides=strides)
        self.conv2 = layers.Conv2D(filters, kernel_size, padding="same",
                                  name="conv2")
        self.bn1 = layers.BatchNormalization(name="bn1")
        self.bn2 = layers.BatchNormalization(name="bn2")

        self.activation1 = layers.Activation(self.activation, name="activation1")
        self.activation2 = layers.Activation(self.activation, name="activation2")

        self.add_shortcut = layers.Add()

        self.conv_shortcut = None
        if self.shortcut_conv:
            self.conv_shortcut = layers.Conv2D(filters, 1, strides=strides,
                                              name="conv_shortcut")

    def call(self, x):
        if self.conv_shortcut is not None:
            shortcut = self.conv_shortcut(x)
        else:
            shortcut = x
        y = self.activation1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.activation2(self.add_shortcut([y, shortcut]))
        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "shortcut_conv": self.shortcut_conv,
            "strides": self.strides,
            "activation": keras.activations.serialize(self.activation),
        })
        return config
