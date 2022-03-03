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

<<<<<<< HEAD:mockmodel.py

#we're defining the discriminator
discriminator = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size = 5, strides =2, padding = 'same' ,activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    #using maxpooling to reduce the resolution when it's too big
    keras.layers.Dropout(0.4), #adding a dropout layer to prevent the neurons from updating the weights only according a specific output
    keras.layers.Conv2D(64, kernel_size = 5, strides =2, padding = 'same' ,activation = 'relu'),
    #adding a conv layer to help produce a tensor of outputs
    #strides: the number of pixerls the kernel is moving by in each direction
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(), #getting a vector representation from flattening the image
    keras.layers.Dense(1, activation = 'sigmoid')]) #final layer

gan = keras.models.Sequential([generator, discriminator])


discriminator.compile(loss='binary_crossentropy',optimizer = 'Adam')
"""using binary crossentropy, the disciminator is a binary classifier"""
discriminator.trainable = False #So that we don’t update the discriminator when updating the generator.
gan.compile(loss='binary_crossentropy',optimizer = 'Adam')
#Adam is a combination of the best of both rmsprop and SGD


def train_gan(gan, dataset, batch_size, codings_size, n_epochs = 50):
=======
def train_gan(gan, dataset, batch_size, codings_size, n_epochs = 1000):
>>>>>>> bacf12128160109aa235c305e0a4bdf6a4de1151:deepOstinato/models/mockmodel.py
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
