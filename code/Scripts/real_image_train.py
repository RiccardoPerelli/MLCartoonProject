import glob
import os
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import PIL
import PIL.Image
from skimage.transform import resize

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

x_train_set = []
x_validation_set = []
x_test_set = []

img_height = 128
img_width = 128
batch_size = 64
epochs = 15

latent_dim = 4096

class RealAutoencoder(Model):
    '''def __init__(self, latent_dim):
              super(Autoencoder, self).__init__()
              self.latent_dim = latent_dim
              self.encoder = tf.keras.Sequential([
                layers.Flatten(),
                layers.Dense(latent_dim, activation='relu'),
              ])
              self.decoder = tf.keras.Sequential([
                layers.Dense(img_height*img_width*3, activation='sigmoid'),
                layers.Reshape((img_height, img_width, 3))
              ])'''

    def __init__(self, latent_dim):
        super(RealAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(img_height, img_width, 3)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Reshape((16, 16, 16)),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(64, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
            layers.Conv2DTranspose(32, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
            layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_real_images():
    train_set = glob.glob('../RealImages/train_cropped/*.jpg')
    # validation_set = glob.glob("../RealImages/val_cropped/*.jpg")
    # test_set = glob.glob("../RealImages/data/test_cropped/*.jpg")

    print("Real autoencoder training...")
    x_train_set = np.array([np.array(PIL.Image.open(fname)) for fname in train_set])
    x_train_set = resize(x_train_set, (len(x_train_set), img_height, img_width, 3))

    realAutoencoder = RealAutoencoder(latent_dim)

    realAutoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    realAutoencoder.fit(x_train_set, x_train_set, epochs=epochs, shuffle=True)
    # validation_data=(validation_set, validation_set))

    realAutoencoder.encoder.summary()
    realAutoencoder.decoder.summary()

    #encoded_real_imgs = realAutoencoder.encoder.predict(x_train_set)
    #decoded_real_imgs = realAutoencoder.decoder.predict(encoded_real_imgs)
    '''
    plt.figure(figsize=(10, 10))
    for i in range(0, 20, 2):
        plt.subplot(4, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(decoded_real_imgs[i], cmap=plt.cm.binary)
        plt.subplot(4, 5, i + 2)
        plt.grid(False)
        plt.imshow(x_train_set[i], cmap=plt.cm.binary)
    plt.show()
    '''

    return realAutoencoder

