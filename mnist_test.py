import tensorflow as tf
from dcgan_model import config, model
import matplotlib.pyplot as plt
import imageio
import glob
import numpy as np
import os

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

    # notice: setup training meta and model.
    global_config = config.globalConfig()
    model_config = config.modelConfig()

    # notice: prepare data.

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = tf.expand_dims(train_images.astype(np.float32), axis=-1) / 255.

    train_datasets = tf.data.Dataset.from_tensor_slices(train_images)

    # shuffle data and set batch size.
    train_datasets = train_datasets.shuffle(train_images.shape[0])
    train_datasets = train_datasets.batch(global_config.batch_size)

    # notice: prepare model.
    dcgan = model.dcgan(model_config)

    # set optimizer.
    generator_optimizer = tf.keras.optimizers.Adam(global_config.learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(global_config.learning_rate)


    # notice: train
    @tf.function
    def step_train(batch):
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            images_generated, d_out_real, d_out_fake = dcgan(images=batch, latent_vector=tf.random.normal(shape=[16, model_config.latent_size]))

            generator_loss = dcgan.generatorLoss(d_out_fake)
            discriminator_loss = dcgan.discriminatorLoss(d_out_fake, d_out_real)


        generator_gradients = generator_tape.gradient(generator_loss, dcgan.generator.trainable_variables)
        discriminator_gradients = discriminator_tape.gradient(discriminator_loss, dcgan.discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, dcgan.generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, dcgan.discriminator.trainable_variables))


    for epx, epoch in enumerate(range(global_config.epochs)):
        print(f'epoch {epx}/{global_config.epochs}')
        for batch in train_datasets:
            step_train(batch)

        # generate.
        generated = dcgan.generator(tf.random.normal(shape=[16, model_config.latent_size]), training=False)
        for idx, i in enumerate(generated):
            plt.subplot(4, 4, idx + 1)
            plt.imshow(generated[idx] * 255., cmap='gray')
            plt.axis('off')

        plt.savefig(f'./dcgan_output/epoch_{epoch}.png')
        plt.show()

    # generate gif;
    anim_file = './dcgan_output/dcgan_mnist.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('./dcgan_output/epoch*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
