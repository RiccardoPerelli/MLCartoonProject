import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import PIL
import PIL.Image
import pathlib
from PIL import ImageOps
from skimage.transform import resize

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import real_image_train as rit
import face_recognition as fr

#============================================#
#========== variable declaration ============#
#============================================#
from tensorflow.python.keras import Input

X_train_set = []
X_validation_set = []
X_test_set = []

#Model input data height and width#
img_height = 128
img_width = 128
batch_size = 32
epochs = 15

#Dimensione del vettore in cui sarà compressa l'immagine#
latent_dim = 8192

train_set = glob.glob("../CartoonImages/data/dummy/*.jpg")
validation_set = glob.glob("../CartoonImages/data/validation/*.jpg")
test_set = glob.glob("../CartoonImages/data/test/*.jpg")

#============================================#
#============== Images Loading ==============#
#============================================#

print("train loading starting:")
x_train_set = np.array([np.array(PIL.Image.open(fname)) for fname in train_set])
x_train_set = resize(x_train_set, (len(x_train_set), img_height, img_width, 3))

#print("validation loading starting:")

#print("test loading starting:")

#train_set = np.array(X_train_set)
#x_train_set = x_train_set.astype('float32') / 255.

#validation_set = np.array(X_validation_set)
#validation_set = validation_set.astype('float32') / 255.

#test_set = np.array(X_test_set)
#test_set = test_set.astype('float32') / 255.

#============================================#
#============= Class declaration ============#
#============================================#
#Classe contenente il modello (l'autoencoder)#
class Autoencoder(Model):

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
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(img_height, img_width, 3)),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2, 2)),
      layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2)),
      layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2)),
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Reshape((8, 8, 128)),
      layers.BatchNormalization(),
      layers.Conv2DTranspose(128, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
      layers.Conv2DTranspose(64, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
      layers.Conv2DTranspose(32, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
      layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train_set, x_train_set,
                epochs=epochs,
                shuffle=True)
                #validation_data=(validation_set, validation_set))

autoencoder.encoder.summary()
autoencoder.decoder.summary()

encoded_imgs = autoencoder.encoder.predict(x_train_set)
decoded_imgs = autoencoder.decoder.predict(encoded_imgs)

#============================================#
#========== Results Visualization ===========#
#============================================#
# Todo: visualizzare immagini ricostruite
'''
plt.figure(figsize=(10,10))
for i in range(0, 20, 2):
    plt.subplot(4,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(decoded_imgs[i], cmap=plt.cm.binary)
    plt.subplot(4, 5, i + 2)
    plt.grid(False)
    plt.imshow(x_train_set[i], cmap=plt.cm.binary)
plt.show()
'''
'''
# IMAGE CROP TEST SET
uncropped_set = glob.glob('../RealImages/test/*.jpg')

for filename in uncropped_set:
    pixels = plt.imread(filename)
    pixels = fr.extract_face(filename)
    plt.imsave('../RealImages/test_cropped/' + os.path.basename(filename), pixels)
'''

rit.train_real_images()