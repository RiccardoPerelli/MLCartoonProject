import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import PIL.Image
from skimage.transform import resize

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input

from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#============================================#
#========== variable declaration ============#
#============================================#
from tensorflow.python.keras import Input

X_cartoon_train_set = []
X_validation_set = []
X_test_set = []

#Model input data height and width#
img_height = 64
img_width = 64
channels = 3
batch_size = 32
epochs = 300

#Dimensione del vettore in cui sarà compressa l'immagine#
latent_dim = 1024

cartoon_train_set = glob.glob("../CartoonImages/data/train/*.jpg")
validation_set = glob.glob("../CartoonImages/data/validation_big/*.jpg")
test_set = glob.glob("../CartoonImages/data/test/*.jpg")

#============================================#
#============== Images Loading ==============#
#============================================#

print("cartoon train loading starting:")
x_cartoon_train_set = np.array([np.array(PIL.Image.open(fname)) for fname in cartoon_train_set])
x_cartoon_train_set = resize(x_cartoon_train_set, (len(x_cartoon_train_set), img_height, img_width, 3))

#============================================#
#============ Common layers def. ============#
#============================================#
#Encoder
inp = Input(shape=(img_width, img_height, channels))
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2))
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', strides=(2, 2))
fc1 = Dense(latent_dim, activation='relu')
fc2 = Dense(latent_dim, activation='relu')

#Decoder
deconv1 = Conv2DTranspose(512, kernel_size=3, strides=(4, 4), activation='relu', padding='same')
deconv2 = Conv2DTranspose(256, kernel_size=3, strides=(2, 2), activation='relu', padding='same')

#print("validation loading starting:")
#print("test loading starting:")
#cartoon_train_set = np.array(X_cartoon_train_set)
#x_cartoon_train_set = x_cartoon_train_set.astype('float32') / 255.
#validation_set = np.array(X_validation_set)
#validation_set = validation_set.astype('float32') / 255.
#test_set = np.array(X_test_set)
#test_set = test_set.astype('float32') / 255.

#============================================#
#============= Class declaration ============#
#============================================#
#Classe contenente il modello (l'autoencoder)#
class CartoonAutoencoder(Model):

  def __init__(self, latent_dim):
    super(CartoonAutoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
        layers.Input(shape=(img_height, img_width, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2)),
        conv3,
        conv4,
        layers.Flatten(),
        fc1,
        fc2
    ])
    self.decoder = tf.keras.Sequential([
        layers.Reshape((1, 1, 1024)),
        layers.BatchNormalization(),
        deconv1,
        deconv2,
        layers.Conv2DTranspose(128, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
        layers.Conv2DTranspose(64, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
        layers.Conv2DTranspose(3, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
        #layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

cartoon_autoencoder = CartoonAutoencoder(latent_dim)

cartoon_autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

cartoon_autoencoder.fit(x_cartoon_train_set, x_cartoon_train_set,
                epochs=epochs,
                shuffle=True)
                #validation_data=(validation_set, validation_set))

cartoon_autoencoder.encoder.summary()
cartoon_autoencoder.decoder.summary()

#encoded_imgs = cartoon_autoencoder.encoder.predict(x_cartoon_train_set)
#decoded_imgs = cartoon_autoencoder.decoder.predict(encoded_imgs)

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
    plt.imshow(x_cartoon_train_set[i], cmap=plt.cm.binary)
plt.show()


encoded_imgs = autoencoder.encoder.predict(x_cartoon_train_set)
decoded_imgs = autoencoder.decoder.predict(encoded_imgs)

#============================================#
#========== Results Visualization ===========#
#============================================#
# Todo: visualizzare immagini ricostruite

plt.figure(figsize=(10,10))
for i in range(0, 20, 2):
    plt.subplot(4,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(decoded_imgs[i], cmap=plt.cm.binary)
    plt.subplot(4, 5, i + 2)
    plt.grid(False)
    plt.imshow(x_cartoon_train_set[i], cmap=plt.cm.binary)
plt.show()

# IMAGE PRE PROCESSING
uncropped_real_train_set = glob.glob('../RealImages/train/*.jpg')
uncropped_test_set = glob.glob('../RealImages/test/*.jpg')
print("Dataset Real Image preprocessing...")
for filename in uncropped_real_train_set:
    if not os.path.isfile('../RealImages/train_cropped' + os.path.basename(filename)):
        image = plt.imread(filename)
        image = dp.portrait_segmentation(filename)
        plt.imsave('../RealImages/train_cropped/' + os.path.basename(filename), image)

for filename in uncropped_test_set:
    if not os.path.isfile('../RealImages/test_cropped' + os.path.basename(filename)):
        image = plt.imread(filename)
        image = dp.portrait_segmentation(filename)
        plt.imsave('../RealImages/test_cropped/' + os.path.basename(filename), image)

for filename in uncropped_set:
    pixels = plt.imread(filename)
    pixels = fr.extract_face(filename)
    plt.imsave('../RealImages/test_cropped/' + os.path.basename(filename), pixels)
'''

class RealAutoencoder(Model):

    def __init__(self, latent_dim):
        super(RealAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(img_height, img_width, 3)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2)),
            conv3,
            conv4,
            layers.Flatten(),
            fc1,
            fc2
        ])
        self.decoder = tf.keras.Sequential([
            layers.Reshape((1, 1, 1024)),
            layers.BatchNormalization(),
            deconv1,
            deconv2,
            layers.Conv2DTranspose(128, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
            layers.Conv2DTranspose(64, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
            layers.Conv2DTranspose(3, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
            # layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

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
real_test_set = glob.glob("../RealImages/test_cropped/*.jpg")

x_real_test_set = np.array([np.array(PIL.Image.open(fname)) for fname in real_test_set])
x_real_test_set = resize(x_real_test_set, (len(x_real_test_set), img_height, img_width, 3))

#============================================#
#=========== Try with real images ===========#
#============================================#
encoded_imgs_final = realAutoencoder.encoder.predict(x_real_test_set)
decoded_imgs_final = cartoon_autoencoder.decoder.predict(encoded_imgs_final)

#============================================#
#========== Results Visualization ===========#
#============================================#
plt.figure(figsize=(10,10))
for i in range(0, 20, 2):
    plt.subplot(4, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_real_test_set[i], cmap=plt.cm.binary)
    plt.subplot(4, 5, i + 2)
    plt.grid(False)
    plt.imshow(decoded_imgs_final[i], cmap=plt.cm.binary)
plt.show()
