from tkinter import Y
import tensorflow as tf
from tensorflow.keras import Sequential, layers
import keras
import numpy as np


"""the number of our features"""
X = ''
Y = ''
Z = ''


generator = keras.models.Sequential([

# the shapes of the layers are trial and error and we can change them based on our data
    keras.layers.Dense(X*Y*Z, input_shape=[512]), #512 to store info in higher dimnesion
    keras.layers.Reshape([X,Y,Z]),
    keras.layers.Conv2DTranspose(32, kernel_size = 3, strides =2, padding = 'same',activation = 'relu'),
    keras.layers.Conv2DTranspose(1, kernel_size = 3, strides =2, padding = 'same',activation = 'relu')
    ])

"""using conv2d to generate an image from a random seed"""
def train_gan(gan, dataset, batch_size, codings_size, n_epochs = 1000):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        for X_batch in dataset:
            """training the disciminator"""
            noise = tf.random.normal(shape =[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis = 0)
            y1= tf.constant([[0.]]* batch_size + [[1.]] * batch_size)
            """seting the target y1 to 0 fo fake images and 1 for real images"""
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            """training the generator"""
            noise = tf.random.normal(shape = [batch_size,codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            """set to False to avoid warning by Keras"""
            gan.train_on_batch(noise.y2)

X_train = ''