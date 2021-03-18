import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import PIL.Image
import os
from skimage.transform import resize
from sklearn.utils import shuffle
#from imutils import build_montages

import matplotlib.pyplot as plt
#from tqdm import tqdm

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

import data_preprocessing as dp
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
batch_size = 256
epochs = 501
alpha = 0.2
INIT_LR=1e-4

#Dimensione del vettore in cui sarà compressa l'immagine#
latent_dim = 1024

leaky = tf.keras.layers.LeakyReLU(alpha)

cartoon_train_set = glob.glob("../CartoonImages/data/dummy/*.jpg")
#validation_set = glob.glob("../CartoonImages/data/validation_big/*.jpg")
#test_set = glob.glob("../CartoonImages/data/test/*.jpg")

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
conv3 = Conv2D(128, (5, 5), activation='relu', padding='same', strides=(1, 1))
conv4 = Conv2D(256, (5, 5), activation='relu', padding='same', strides=(1, 1))
fc1 = Dense(latent_dim, activation='relu')
fc2 = Dense(latent_dim, activation='relu')

#Decoder
deconv1 = Conv2DTranspose(512, kernel_size=5, strides=(4, 4), activation='relu', padding='same')
deconv2 = Conv2DTranspose(256, kernel_size=4, strides=(2, 2), activation='relu', padding='same')

#plottare il modello
#tf.keras.utils.plot_model(model, "simple_resnet.png",show_shapes=True)

def build_cartoon_decoder(width, height, inputDim=1024, n1=512, channels=3):

    inp = Input(shape=(inputDim,))
    
    # FC - BN
    dim1 = 1
    dim2 = 1
    
    x = Dense(dim1 * dim2 * inputDim, activation="relu")(inp)
    
    x = Reshape((dim1, dim2, inputDim))(x)
    
    x = deconv1(x)
    x = BatchNormalization()(x)
    x = deconv2(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(128, kernel_size=4, strides=(2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, kernel_size=4, strides=(2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    out = Conv2DTranspose(3, (3, 3), strides=(2, 2), activation='tanh', padding='same')(x)
    
    m = Model(inputs=inp, outputs=out)
    return m
    
def build_real_decoder(width, height, inputDim=1024, n1=512, channels=3):

    inp = Input(shape=(inputDim,))
    
    # FC - BN
    dim1 = 1
    dim2 = 1
    
    y = Dense(dim1 * dim2 * n1, activation="relu")(inp)
    
    y = Reshape((dim1, dim2, n1))(y)
    
    y = deconv1(y)
    y = BatchNormalization()(y)
    y = deconv2(y)
    y = BatchNormalization()(y)
    y = Conv2DTranspose(128, kernel_size=4, strides=(2, 2), activation='relu', padding='same')(y)
    y = BatchNormalization()(y)
    y = Conv2DTranspose(64, kernel_size=4, strides=(2, 2), activation='relu', padding='same')(y)
    y = BatchNormalization()(y)
    out = Conv2DTranspose(3, (3, 3), strides=(1, 1), activation='tanh', padding='same')(y)
    
    m = Model(inputs=inp, outputs=out)
    return m
    
cartoon_decoder = build_cartoon_decoder(img_height, img_width, latent_dim, latent_dim, channels)
cartoon_decoder.summary()

real_decoder = build_real_decoder(img_height, img_width, latent_dim, latent_dim, channels)
real_decoder.summary()

def build_cartoon_encoder(width, height, channels, alpha=0.2, droprate=0.4):

    input_shape = (width, height, channels)
    
    inp = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), activation=leaky, padding='same', strides=(1, 1))(inp)
    x = MaxPooling2D(strides=2)(x)
    x = Conv2D(64, (5, 5), activation=leaky, padding='same', strides=(1, 1))(x)
    x = MaxPooling2D(strides=2)(x)
    x = conv3(x)
    x = MaxPooling2D(strides=2)(x)
    x = conv4(x)
    x = MaxPooling2D(strides=2)(x)
    x = Flatten()(x)
    x = fc1(x)
    out = fc2(x)
    
    m = Model(inputs=inp, outputs=out)
    return m
    
def build_real_encoder(width, height, channels, alpha=0.2, droprate=0.4):

    input_shape = (width, height, channels)
    
    inp = Input(shape=input_shape)
    y = Conv2D(32, (5, 5), activation=leaky, padding='same', strides=(1, 1))(inp)
    y = MaxPooling2D(strides=2)(y)
    y = Conv2D(64, (5, 5), activation=leaky, padding='same', strides=(1, 1))(y)
    y = MaxPooling2D(strides=2)(y)
    y = conv3(y)
    y = MaxPooling2D(strides=2)(y)
    y = conv4(y)
    y = MaxPooling2D(strides=2)(y)
    y = Flatten()(y)
    y = fc1(y)
    out = fc2(y)
    
    m = Model(inputs=inp, outputs=out)
    return m
    
cartoon_encoder = build_cartoon_encoder(img_height, img_width, 3, droprate=0.5)
cartoon_encoder.summary()

real_encoder = build_real_encoder(img_height, img_width, 3, droprate=0.5)
real_encoder.summary()

cartoon_encoder_optimizer = tf.keras.optimizers.Adam(lr=INIT_LR, beta_1=0.5)
cartoon_encoder.compile(cartoon_encoder_optimizer, loss="binary_crossentropy", metrics=["accuracy"])
cartoon_decoder.compile(cartoon_encoder_optimizer, loss="binary_crossentropy", metrics=["accuracy"])
cartoon_encoder.summary()

real_encoder_optimizer = tf.keras.optimizers.Adam(lr=INIT_LR, beta_1=0.5)
real_encoder.compile(real_encoder_optimizer, loss="binary_crossentropy", metrics=["accuracy"])
real_decoder.compile(real_encoder_optimizer, loss="binary_crossentropy", metrics=["accuracy"])
real_encoder.summary()

cartoon_autoencoder = tf.keras.Sequential([cartoon_encoder, cartoon_decoder])
real_autoencoder = tf.keras.Sequential([real_encoder, real_decoder])

autoencoderOpt = tf.keras.optimizers.Adam(lr=INIT_LR, beta_1=0.5)

cartoon_autoencoder.compile(loss="mse", optimizer=autoencoderOpt, metrics=["accuracy"])
cartoon_autoencoder.summary()

real_autoencoder.compile(loss="mse", optimizer=autoencoderOpt)
real_autoencoder.summary()

def preprocessing_function(x):
  return (x - 127.5)/127.5


def plot_figures(x, n, figsize=None):
    if figsize:
        plt.figure(figsize=figsize)
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img = x[i, :, :, :]
        # rescale for visualization purposes
        img = ((img * 127.5) + 127.5).astype("uint8")
        plt.imshow(img)

    plt.show()

batchPerEpoch = int(2000 / batch_size)

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=0,
    shear_range=0,
    zoom_range=0,
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=preprocessing_function
)

history = {}

history['G_loss'] = []
history['D_loss_true'] = []
history['D_loss_fake'] = []
accuracy = {}
accuracy['Acc_true'] = []
accuracy['Acc_fake'] = []

def generate_latent_points(latent_dim, n_samples):

	x_input = np.random.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input


cartoon_train = train_generator.flow_from_directory("../CartoonImages/data/train", target_size=(64,64), batch_size=batch_size)
#cartoon_train = train_generator.flow_from_directory("../CartoonImages/data/train", target_size=(64,64), batch_size=batch_size)
real_train = train_generator.flow_from_directory("../RealImages/train_cropped", target_size=(64,64,3), batch_size=batch_size)

for epoch in range(epochs):
    for b in range(batchPerEpoch):
        # now train the discriminator to differentiate between true and fake images
        trueImages, _ = next(cartoon_train)
        # one sided label smoothing reduces overconfidence in true images and stabilizes training a bit
        '''
        y = 0.9 * np.ones((trueImages.shape[0]))
        discLoss, discAcc = cartoon_encoder.train_on_batch(trueImages, y)
        history['D_loss_true'].append(discLoss)
        # warning: accuracy will not be calculated if label smoothing is used
        accuracy['Acc_true'].append(discAcc)

        # generate some fake samples
        noise = generate_latent_points(latent_dim, batch_size)
        genImages = cartoon_decoder.predict(noise)
        y = np.zeros((batch_size))

        discLoss, discAcc = cartoon_encoder.train_on_batch(genImages, y)
        history['D_loss_fake'].append(discLoss)
        accuracy['Acc_fake'].append(discAcc)
        '''
        # mixing true and fake samples (like below) may prevent training, especially when using batch normalization
        # X =  np.concatenate((trueImages, genImages))
        # y = np.concatenate((np.ones((BATCH_SIZE_2)), np.zeros((BATCH_SIZE_2))))
        # (X, y) = shuffle(X, y)
        # discLoss, discAcc = disc.train_on_batch(X, y)
        # history['D_loss'].append(discLoss)

        # now train the generator
        #noise = generate_latent_points(latent_dim, batch_size)
        # some authors suggest randomly flipping some labels to introduce random variations
        #fake_labels = [1] * batch_size
        #fake_labels = np.reshape(fake_labels, (-1,))
        #ganLoss = cartoon_autoencoder.train_on_batch(noise, fake_labels)
        #history['G_loss'].append(ganLoss)
        #y = 0.9 * np.ones((trueImages.shape[0]))
        cartoon_autoencoder_Loss, cartoon_autoencoder_Acc = cartoon_autoencoder.train_on_batch(trueImages, trueImages)
        # at the end of each epoc
    print("epoch " + str(epoch) + ": autoencoder loss " + str(cartoon_autoencoder_Loss) + " ( " + str(
        cartoon_autoencoder_Acc) + " ) ")

    # it is important to regularly visualize the output
    images = cartoon_autoencoder.predict(trueImages)
    #images = gen.predict(benchmarkNoise)
    if (epoch % 50) == 0:
        plt.figure(figsize=(10, 10))
        for i in range(0, 20, 2):
            plt.subplot(4, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i], cmap=plt.cm.binary)
            plt.subplot(4, 5, i + 2)
            plt.grid(False)
            plt.imshow(trueImages[i], cmap=plt.cm.binary)
        plt.show()

    # save regularly as the training may destabilize and start producing garbage image again
    #if (epoch % 100) == 0:
        #gan.save(os.path.join(directory_models, "model64_0103" + str(epoch) + ".h5"))

'''
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
        layers.Conv2D(32, (5, 5), activation='relu', padding='same', strides=(2, 2)),
        layers.Conv2D(64, (5, 5), activation='relu', padding='same', strides=(2, 2)),
        conv3,
        conv4,
        layers.Flatten(),
        fc1,
        fc2
    ])
    self.decoder = tf.keras.Sequential([
        layers.Reshape((1, 1, 1024)),
        deconv1,
        layers.BatchNormalization(),
        deconv2,
        layers.BatchNormalization(),
        layers.Conv2DTranspose(128, kernel_size=5, strides=(2, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(64, kernel_size=5, strides=(2, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(3, kernel_size=5, strides=(2, 2), activation='tanh', padding='same'),
        #layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

cartoon_autoencoder = CartoonAutoencoder(latent_dim)

INIT_LR=2e-4
# Adam with lower values of beta_1 was found to achieve best performances for different GANs architecture
# GANs are quite sensitive to the learning rate
discOpt = tf.keras.optimizers.Adam(lr=INIT_LR, beta_1=0.5)

cartoon_autoencoder.compile(discOpt, loss=losses.MeanSquaredError())

cartoon_autoencoder.fit(x_cartoon_train_set, x_cartoon_train_set,
                epochs=epochs,
                shuffle=True)
                #validation_data=(validation_set, validation_set))

cartoon_autoencoder.encoder.summary()
cartoon_autoencoder.decoder.summary()

encoded_imgs = cartoon_autoencoder.encoder.predict(x_cartoon_train_set)
decoded_imgs = cartoon_autoencoder.decoder.predict(encoded_imgs)

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

'''

'''
# IMAGE PRE PROCESSING
uncropped_real_all_set = glob.glob('../RealImages/all/*.jpg')
uncropped_real_train_set = glob.glob('../RealImages/train/*.jpg')
print("Dataset Real Image preprocessing...")
for filename in uncropped_real_all_set:
    if not os.path.isfile('../RealImages/all_cropped' + os.path.basename(filename)):
        image = plt.imread(filename)
        image = dp.portrait_segmentation(filename)
        plt.imsave('../RealImages/all_cropped/' + os.path.basename(filename), image)

for filename in uncropped_real_train_set:
    if not os.path.isfile('../RealImages/train_cropped' + os.path.basename(filename)):
        image = plt.imread(filename)
        image = dp.portrait_segmentation(filename)
        plt.imsave('../RealImages/train_cropped/' + os.path.basename(filename), image)

for filename in uncropped_set:
    pixels = plt.imread(filename)
    pixels = fr.extract_face(filename)
    plt.imsave('../RealImages/test_cropped/' + os.path.basename(filename), pixels)
'''

'''
epochs = 200
class RealAutoencoder(Model):

    def __init__(self, latent_dim):
        super(RealAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(img_height, img_width, 3)),
            layers.Conv2D(32, (5, 5), activation='relu', padding='same', strides=(2, 2)),
            layers.Conv2D(64, (5, 5), activation='relu', padding='same', strides=(2, 2)),
            conv3,
            conv4,
            layers.Flatten(),
            fc1,
            fc2
        ])
        self.decoder = tf.keras.Sequential([
            layers.Reshape((1, 1, 1024)),
            deconv1,
            layers.BatchNormalization(),
            deconv2,
            layers.BatchNormalization(),
            layers.Conv2DTranspose(128, kernel_size=5, strides=(2, 2), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(64, kernel_size=5, strides=(2, 2), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(3, kernel_size=5, strides=(2, 2), activation='tanh', padding='same'),
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

realAutoencoder.compile(discOpt, loss=losses.MeanSquaredError())

realAutoencoder.fit(x_train_set, x_train_set, epochs=epochs, shuffle=True)
# validation_data=(validation_set, validation_set))

realAutoencoder.encoder.summary()
realAutoencoder.decoder.summary()

#encoded_real_imgs = realAutoencoder.encoder.predict(x_train_set)
#decoded_real_imgs = realAutoencoder.decoder.predict(encoded_real_imgs)
'''

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

'''
