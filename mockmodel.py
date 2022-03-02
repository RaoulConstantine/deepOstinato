import tensorflow as tf
from tensorflow.keras import Sequential, layers
import keras
import numpy as np


generator = keras.models.Sequential([
    keras.layers.Dense(1024*2930*1, input_shape=[100]),
    keras.layers.Reshape([1024,2930,1]),
    keras.layers.Conv2DTranspose(64, kernel_size = 5, strides =2, padding = 'same',activation = 'relu'),
    keras.layers.Conv2DTranspose(64, kernel_size = 5, strides =2, padding = 'same',activation = 'relu')])

discriminator = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size = 5, strides =2, padding = 'same' ,activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(64, kernel_size = 5, strides =2, padding = 'same' ,activation = 'relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation = 'sigmoid')])

gan = keras.models.Sequential([generator, discriminator])


discriminator.compile(loss='binary_crossentropy',optimizer = 'rmsprop')
discriminator.trainable = False #So that we don’t update the discriminator when updating the generator.
gan.compile(loss='binary_crossentropy',optimizer = 'rmsprop')

def train_gan(gan, dataset, batch_size, codings_size, n_epochs = 50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        for X_batch in dataset:
            noise = tf.random.normal(shape =[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis = 0)
            y1= tf.constant([[0.]]* batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            noise = tf.random.normal(shape = [batch_size,codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise.y2)

X_train = np.load('/content/drive/MyDrive/audio/npy_file_no0.npy').reshape((1024,2930,1)) #loading from g_drive