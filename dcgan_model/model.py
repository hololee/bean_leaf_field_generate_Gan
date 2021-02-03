import tensorflow as tf


class dcgan():
    # Notice: generator.
    class Generator(tf.keras.Model):
        def __init__(self, configure):
            super().__init__()
            self.configure = configure

            self.g_dense1 = tf.keras.layers.Dense(7 * 7 * 256, use_bias=True, input_shape=(self.configure.latent_size,))
            self.g_dense1_batch = tf.keras.layers.BatchNormalization()
            self.g_dense1_activation = tf.keras.layers.LeakyReLU()

            self.g_reshape = tf.keras.layers.Reshape((7, 7, 256))

            self.g_conv1 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
            self.g_conv1_batch = tf.keras.layers.BatchNormalization()
            self.g_conv1_activation = tf.keras.layers.LeakyReLU()

            self.g_conv2 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
            self.g_conv2_batch = tf.keras.layers.BatchNormalization()
            self.g_conv2_activation = tf.keras.layers.LeakyReLU()

            self.g_conv3 = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False)
            self.g_conv3_activation = tf.keras.activations.tanh

        def __call__(self, inputs, training=False):
            g_out = self.g_dense1(inputs)
            g_out = self.g_dense1_batch(g_out, training=training)
            g_out = self.g_dense1_activation(g_out)

            g_out = self.g_reshape(g_out)

            g_out = self.g_conv1(g_out)
            g_out = self.g_conv1_batch(g_out, training=training)
            g_out = self.g_conv1_activation(g_out)
            g_out = self.g_conv2(g_out)
            g_out = self.g_conv2_batch(g_out, training=training)
            g_out = self.g_conv2_activation(g_out)
            g_out = self.g_conv3(g_out)
            g_out = self.g_conv3_activation(g_out)

            return g_out

    # Notice: Discriminator.
    class Discriminator(tf.keras.Model):
        def __init__(self, configure):
            super().__init__()
            self.configure = configure

            self.d_conv1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1])
            self.d_conv1_activation = tf.keras.layers.LeakyReLU()
            self.d_conv1_drop = tf.keras.layers.Dropout(self.configure.drop_rate)

            self.d_conv2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
            self.d_conv2_activation = tf.keras.layers.LeakyReLU()
            self.d_conv2_drop = tf.keras.layers.Dropout(self.configure.drop_rate)

            self.d_flatten = tf.keras.layers.Flatten()
            self.d_dense_last = tf.keras.layers.Dense(1)

        def __call__(self, inputs, training=False):
            d_out = self.d_conv1(inputs)
            d_out = self.d_conv1_activation(d_out)
            d_out = self.d_conv1_drop(d_out, training=training)

            d_out = self.d_conv2(d_out)
            d_out = self.d_conv2_activation(d_out)
            d_out = self.d_conv2_drop(d_out, training=training)

            d_out = self.d_flatten(d_out)
            d_out = self.d_dense_last(d_out)

            return d_out

    def __init__(self, configure):
        super(dcgan, self).__init__()
        self.configure = configure
        self.generator = self.Generator(self.configure)
        self.discriminator = self.Discriminator(self.configure)

    def generatorLoss(self, d_out_fake):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(d_out_fake), d_out_fake)

    def discriminatorLoss(self, d_out_fake, d_out_real):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(d_out_real), d_out_real)
        fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(d_out_fake), d_out_fake)

        return real_loss + fake_loss

    def __call__(self, images, latent_vector):
        images_generated = self.generator(latent_vector, training=True)
        d_out_real = self.discriminator(images, training=True)
        d_out_fake = self.discriminator(images_generated, training=True)

        return images_generated, d_out_real, d_out_fake
