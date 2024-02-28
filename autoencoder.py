""""======Modules======="""
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input , Conv2D , Conv2DTranspose , MaxPooling2D , UpSampling2D   # model layers
)
from tensorflow.keras.models import Model                            # functional model
from tensorflow.keras.losses import MeanSquaredError                 # loss metric
from tensorflow.keras.optimizers import Adam 
from keras import layers
from keras.datasets import mnist
from keras.models import Model, load_model
import scipy

"""====Functions===="""
def split_data(im_fold, seed_nb):
    """
    Proceed to the splitting of the data in a train set and a validation set

    Parameters :
        im_fold (string) : path to the folder containing the pictures
        seed_nb (int) : number of the seed you want to use

    Return :
        train_data (keras.src.preprocessing.image.DirectoryIterator): set of the images used to train
        val_data (keras.src.preprocessing.image.DirectoryIterator): set of the images used for validation
    """
    train_augment=ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        zoom_range=0.1,
        shear_range=0.1,
        validation_split=0.2,
    )
    val_augment=ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    train_data=train_augment.flow_from_directory(
        im_fold,
        target_size=(218,178),
        batch_size=20,
        subset='training',
        class_mode='input',
        seed=seed_nb
    )
    val_data=val_augment.flow_from_directory(
        im_fold,
        target_size=(218,178),
        batch_size=20,
        subset='validation',
        class_mode='input',
        seed=seed_nb
    )
    return train_data, val_data

def display_data_set(data):
    """
    Display the 8th first pictures of the 1rst batch of a set of data

    Parameters : 
        data (numpy array) : set of image data (train or validation set)
    
    """
    plt.figure(figsize=(10, 10))
    for image in data[0]:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image[i])
            plt.axis("off")
        plt.show()
        break

"""====Tests===="""
print("Proceed to split data :")
folder="./data/small_set"
train_data, val_data=split_data(folder, seed_nb=40)
print("Test images loaded in train data : ")
display_data_set(train_data)
print('Datatype of train data : ', type(train_data))
print("Test images loaded in val data : ")
display_data_set(val_data)