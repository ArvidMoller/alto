import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt

import torch

os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
from keras.preprocessing.image import img_to_array, array_to_img, load_img

import io
# import imageio
from IPython.display import Image, display
# from ipywidgets import widgets, Layout, HBox

keras.mixed_precision.set_global_policy("mixed_float16")
print("Pytorch version. ", torch.__version__, "GPU name: ", torch.cuda.get_device_name())


# Imports images and converts them from .png to numpy arrays containing 1 channel images. Those images are then added to a sequence array containing the number of images specified in the sequence_size. The sequence array is then appended to a list and converted to a 5-dimensional numpy array. Sequences containing missing images (as specified in ../satellite_imagery_download/images/downlog_log.txt) as skipped.
# 
# Parameters:
# path: Path to images.
# sequence_size: Size of image sequences (time-steps).
#
# Returns:
# dataset: A 5 dimensional numpy array containing all training images of shape (samples, sequence_size (aka time), height, width, channels). 
def load_dataset(path, sequence_size):
    dataset = []
    missing_imgs = []
    missing = False
    
    sample_size = len(os.listdir(path))
    print(f"Total number of pictures: {sample_size}")

    sample_size = int(input("Number of training pictures: "))

    file = open("../satellite_imagery_download/images/download_log.txt", "r")
    lines = file.readlines()
    for i in lines:
        missing_imgs.append(dt.datetime.fromisoformat(i.strip()[:23]))
    file.close()

    print(missing_imgs, "\n")

    for j in range(0, sample_size - (sequence_size-1)):   # loops through all images up to and including the image sequence_size places from the last.
        sequence = []

        first_img = f"{path}/{os.listdir(path)[j]}"[44:][:23]
        for i in [13, 16]:
            first_img = first_img[:i] + ":" + first_img[i+1:]
        
        first_img_dt = dt.datetime.fromisoformat(first_img)
        last_img_dt =  first_img_dt + dt.timedelta(minutes=15*(sequence_size-1))

        print(f"First img as dt: {first_img_dt}, last img as dt: {last_img_dt}")

        for i in missing_imgs:
            if first_img_dt <= i <= last_img_dt:
                missing = True
                print(f"Missing img: {i}")
            else:
                missing = False

        if missing != True:
            for e in range(0, sequence_size):       # loops through the pictures for the next sequence.
                img = load_img(f"{path}/{os.listdir(path)[j + e]}")     # loads images as a PIL image
                img = img.convert("L")      # Converts images to "true gray-scale" (1 channel)
                img_arr = img_to_array(img)     # converts images to numpy arrays
                print(img_arr.shape, os.listdir(path)[j+e], j+e)
                sequence.append(img_arr)     # adds image array to python list
        
            sample = np.stack(sequence, axis=0)     # creates a 4 dimensional numpy array from the python list containing img_arr 
            dataset.append(sample)      # adds sample array to python list

        print("\n")

    dataset = np.stack(dataset, axis=0)     # creates a 5 dimensional numpy array from the python list of arrays

    print(dataset.shape)

    return dataset


# Splits the datset into a training dataset and a validation dataset using indexing and shuffeling the dataset bofore splitting.
#
# Parameters:
# dataset: The numpy array to split.
# split_procentage: Proentage of the dataset that becomes the training dataset.
#
# Returns:
# train_dataset: Dataset for training.
# val_dataset: Dataset for validation.
def split_dataset(dataset, split_procentage):
    indexes = np.arange(len(dataset))
    np.random.shuffle(indexes)
    train_indexes, val_indexes = indexes[: int((len(indexes)*split_procentage))], indexes[int((len(indexes)*split_procentage)) :]
    train_dataset, val_dataset = dataset[train_indexes], dataset[val_indexes]

    return train_dataset, val_dataset


# Shifts elements in an array in order to get an 'x' and 'y' array. 
#
# Parameters:
# data: Array containing the training data.
#
# Returns: 
# x: Array containing frames 0 thourgh n-1, where n is the total number of frames in each sequence.
# y: Array containing frames 1 thourgh n, where n is the total number of frames in each sequence.
def shift_frames(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    return x, y


def construct_model(x_train):
    # Construct the input layer with no definite frame size.
    inp = layers.Input(shape=(None, *x_train.shape[2:]))

    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = layers.ConvLSTM2D(
        filters=32,                     # few filters
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
    )(x)

    # Next, we will build the complete model and compile it.
    model = keras.models.Model(inp, x)

    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(),
    )

    return model


def load_model_for_training(path):
    name = input("What is the name of the model you want to load? ")
    model = keras.saving.load_model(f"{path}/{name}_checkpoint.keras")

    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(),
    )

    return model


# Saves the model as .keras file.
#
# Parameters:
# model: The model object.
# path: The path the file should be saved at. 
# name: The name of the saved model.
#
# Returns: 
# void
def save_model(model, path, name, epochs, batch_size, dataset_shape):
    model.save(f"{path}/{name}.keras", overwrite=False, zipped=True)
    
    with open(f"{path}/{name}_info.txt", "w") as f:
        f.write(f"{input("General info about model: ")}\n\nDataset shape: {dataset_shape}\nEpochs: {epochs}\nBatch size: {batch_size}\nFirst layer filters: {input("Number of filters in first layer: ")}\nSecond layer filters: {input("Number of filters in second layer: ")}\nThird layer filters: {input("Number of filters in third layer: ")}\n3D layer filters: {input("Number of filters in 3D layer: ")}\nFirst layer kernel size: {input("Kernel size in first layer: ")}\nSecond layer kernel size: {input("Kernel size in second layer: ")}\nThird layer kernel size: {input("Kernel size in third layer: ")}\n3D layer kernel size: {input("Kernel size in 3D layer: ")}")


#
# PREPARE DATASET
#

model_name = input("Name of model: ")


dataset = load_dataset("../satellite_imagery_download/images/images", 10)
# print(dataset.shape)

train_dataset, val_dataset = split_dataset(dataset, 0.9)


# Normalize the data to the 0-1 range.
train_dataset = train_dataset / 255
val_dataset = val_dataset / 255

# Apply the processing function to the datasets.
x_train, y_train = shift_frames(train_dataset)
x_val, y_val = shift_frames(val_dataset)

# Inspect the dataset.
print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))


#
#  MODEL TRAINING
#

if input("Should training continue on last checkpoint? (y/n) ") == "n":
    model = construct_model(x_train)
else:
    model = load_model_for_training("../models/checkpoints")
    

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# Define modifiable training hyperparameters.
epochs = 20
batch_size = 5

checkpoint_path = f"../models/checkpoints/{model_name}_checkpoint.keras"
model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    verbose=1,
    save_best_only=False
    )


# Fit the model to the training data.
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    steps_per_epoch=None,       # default is sample size divided with epochs
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

save_model(model, "../models", model_name, epochs, batch_size, dataset.shape)