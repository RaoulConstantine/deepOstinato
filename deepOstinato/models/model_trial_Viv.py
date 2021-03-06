import tensorflow as tf
from tensorflow.keras import Sequential, layers
import numpy as np

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
#from deepOstinato.preprocessing.loader import load_npy_audio
#from keras.optimizers import Adam

class GAN():
    def __init__(self):
        """Spectrogram dimensions"""
        self.spec_rows = 32
        self.spec_cols = 256
        self.channels = 64
        self.spec_shape = (self.spec_rows, self.spec_cols, self.channels)

        self.noise_shape = 100

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                                optimizer="Adam",
                                                metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer="Adam")

        # Spectrograms generated by Generator with random input

        self.generator_input = Input(shape=(self.noise_shape,))
        spec = self.generator(self.generator_input)

        # Switch to turn on or off the training of the discriminator
        self.discriminator.trainable = False
        valid = self.discriminator(spec) #discriminator prediction

        # GAN model  (combination of generator input and discriminator ouput), optimized on the discriminator validity check
        self.combined = Model(self.generator_input, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer="Adam", metrics=["accuracy"])

        # Training parameters
        self.batch_size = 128
        self.epochs = 200

    def build_generator(self):

        model = Sequential([
                        layers.Dense(self.spec_rows*self.spec_cols*self.channels, use_bias=False, input_shape=(None, self.noise_shape)),
                        layers.Reshape([self.spec_rows,self.spec_cols,self.channels]),
                        layers.BatchNormalization(),
                        layers.ReLU(),

                        layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation = 'relu'),
                        layers.Reshape([self.spec_rows,self.spec_cols*2,self.channels]),
                        layers.BatchNormalization(),
                        layers.ReLU(),

                        layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation = 'relu'),
                        layers.Reshape([64*16,1024,1]),
                        layers.ReLU(),
        ])

        return model

    def build_discriminator(self):

        spec_shape = (self.spec_rows, self.spec_cols, self.channels)

        model = Sequential([
            layers.Conv2D(64, kernel_size = 5, strides =2, padding = 'same' ,activation = 'relu'),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Dropout(0.4),
            layers.Conv2D(64, kernel_size = 5, strides =2, padding = 'same' ,activation = 'relu'),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(1, activation = 'sigmoid')
        ])

        return model
        #spec = Input(shape=spec_shape)
        #validity = model(spec)
        #return Model(spec, validity)

    def build_gan(self, generator, discriminator):
        gan_model = Sequential([generator, discriminator])
        return gan_model

    def train_discriminator(self, X_train, batch_size):

        valid_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        #Train on real spectrograms
                #define index of batch to take from X_train, and batch
        batch_index = np.random.randint(0, X_train.shape[0], batch_size)
        valid_specs = X_train[batch_index]
                #train on batch
        self.discriminator.train_on_batch(valid_specs, valid_labels)

        #Train on fake spectrograms
                #define index of batch to take from X_train, and batch
        noise = np.random.normal(0, 1, (batch_size, self.noise_shape))
        fake_specs = self.generator.predict(noise)
                #train on batch
        self.discriminator.train_on_batch(fake_specs, fake_labels)

    def train_generator(self, batch_size):
        valid_labels = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, self.noise_shape))
        self.model.train_on_batch(noise, valid_labels)


    def train_gan(self, path):
        epochs=self.epochs
        batch_size=self.batch_size

        # Load the dataset
        audios = np.array(load_npy_audio(path))
        X_train = np.expand_dims(audios, axis=3)

        #Build GAN model
        generator = self.build_generator()
        discriminator = self.build_discriminator()
        gan_model = self.build_gan(generator, discriminator)

        # Preprocessing
        for epoch in range(self.epochs):
            pass

            # Select a random "real" spectrograms

            #---load method

            # Generate a new spectrograms
            #generated_specs = self.generator.predict(noise)

            # Train the discriminator

            # Train the generator

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def train_gan(gan, dataset, batch_size=32, codings_size = 100, n_epochs = 500):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print(epoch)

        """training the disciminator"""
        noise = np.random.normal(0, 1, (batch_size, 100))
        generated_images = generator(noise)
        print(generated_images.shape)
        X_fake_and_real = tf.concat([generated_images, dataset], axis = 0)
        y1= tf.constant([[0.]]* batch_size + [[1.]] * batch_size)

        """setting the target y1 to 0 fo fake images and 1 for real images"""
        discriminator.trainable = True
        y1 = np.array(y1)
        discriminator.train_on_batch(X_fake_and_real, y1)

        """training the generator"""
        noise = tf.random.normal(shape = [batch_size,codings_size])
        y2 = tf.constant([[1.]] * batch_size)
        discriminator.trainable = False

        """set to False to avoid warning by Keras"""
        print(y1)
        gan.train_on_batch(noise,y2)


#if __name__ == '__main__':
 #   gan = GAN()
  #  gan.train(epochs=30000, batch_size=32)
