import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import PIL
import PIL.Image
import pathlib

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

img_height = 28
img_width = 28
batch_size = 2

model = tf.keras.Sequential([
    layers.Input((28, 28, 1)),
    layers.Conv2D(16, 3, padding='same'),
    layers.Conv2D(32, 3, padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10)
])


# Using dataset_from_directory #
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "D:/Riccardo/University/MachineLearning/Project/repo/MLCartoonProject/code/CartoonImages/",
    labels='inferred',
    label_mode="int",
    #class_names=['cartoonFace'],
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training",
)

# Using dataset_from_directory #
ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'D:/Riccardo/University/MachineLearning/Project/repo/MLCartoonProject/code/CartoonImages/',
    labels='inferred',
    label_mode="int",
    #class_names=['cartoonFace'],
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation",
)

print(ds_train.shape)
print(ds_validation.shape)