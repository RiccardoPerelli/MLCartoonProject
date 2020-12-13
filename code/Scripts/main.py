import glob
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#============================================#
#========== variable declaration ============#
#============================================#
X_train_set = []
X_validation_set = []
X_test_set = []

#Model input data height and width#
img_height = 28
img_width = 28
batch_size = 32

#Dimensione del vettore in cui sarà compressa l'immagine#
latent_dim = 64

train_set = glob.glob("../CartoonImages/data/train/*.png")
validation_set = glob.glob("../CartoonImages/data/validation/*.png")
test_set = glob.glob("../CartoonImages/data/test/*.png")

#============================================#
#============== Images Loading ==============#
#============================================#
for element in train_set:
    image = ImageOps.grayscale(PIL.Image.open(element))
    image = np.asarray(image)
    image_resized = resize(image, (28, 28))
    X_train_set.append(image_resized)

for element in validation_set:
    image = ImageOps.grayscale(PIL.Image.open(element))
    image = np.asarray(image)
    image_resized = resize(image, (28, 28))
    X_validation_set.append(image_resized)

for element in test_set:
    image = ImageOps.grayscale(PIL.Image.open(element))
    image = np.asarray(image)
    image_resized = resize(image, (28, 28))
    X_test_set.append(image_resized)

train_set = np.array(X_train_set)
train_set = train_set.astype('float32') / 255.

validation_set = np.array(X_validation_set)
validation_set = validation_set.astype('float32') / 255.

test_set = np.array(X_test_set)
test_set = test_set.astype('float32') / 255.


#============================================#
#============= Class declaration ============#
#============================================#
#Classe contenente il modello (l'autoencoder)#
class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(784, activation='sigmoid'),
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(train_set, train_set,
                epochs=10,
                shuffle=True,
                validation_data=(validation_set, validation_set))

#============================================#
#============== What's next? ================#
#============================================#
# Todo: visualizzare immagini ricostruite  
