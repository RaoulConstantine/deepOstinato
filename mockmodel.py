import tensorflow
from tensorflow import Sequential, layers


def generator():
    model = Sequential()
    model.add(layers.Dense(3000*256*64, input_shape=[512])),
    """adding a dense layer with a given dimension"""
    model(layers.Reshape([3000,256,64])),
    """reshaping the layerto get a tensor"""
    model(layers.Conv2DTranspose(64, kernel_size = 5, strides =2, padding = 'same',activation = 'relu')),
    """feeding the tensor to a convolutional layer"""
    #check if batch normalization is needed
    model(layers.Conv2DTranspose(64, kernel_size = 5, strides =2, padding = 'same',activation = 'relu')),

def discriminator():
    model = Sequential()
    model(layers.Conv2D(64, kernel_size = 5, strides =2, padding = 'same' ,activation = 'relu')),
    model(layers.Dropout(0.4)),
    model(layers.Conv2D(64, kernel_size = 5, strides =2, padding = 'same' ,activation = 'relu')),
    model(layers.Dropout(0.4)),
    model(layers.Flatten()),
    model.add(layers.Dense(1, activation = 'sigmoid'))

model = Sequential()
gan = model(generator, discriminator)

model.compile(loss='binary_crossentropy', optimizer = 'rmsprop')
discriminator.trainable = False #So that we don’t update the discriminator when updating the generator.


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
