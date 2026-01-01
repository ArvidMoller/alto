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
# start_index: The index where the download starts.
# time_delta: Time between images (most often 15).
#
# Returns:
# dataset: A 5 dimensional numpy array containing all training images of shape (samples, sequence_size, height, width, channels). 
def load_dataset(path, sequence_size, start_index, time_delta):
    dataset = []
    missing_imgs =[]

    print(f"Total number of pictures: {len(os.listdir(path))}")
    sample_size = int(input("Number of training pictures: "))

    first_img = os.listdir(path)[start_index][:23]      # Get file name of first image in download.
    
    for i in [13, 16]:      # Convert to ISO-format.
            first_img = first_img[:i] + ":" + first_img[i+1:]

    first_img_dt = dt.datetime.fromisoformat(first_img)     # Convert to datetime format.

    file = open("../satellite_imagery_download/images/download_log.txt", "r")       # Open download_log.txt and add all missing timestamps to an array.
    lines = file.readlines()
    for i in lines:
        missing_imgs.append(i.strip()[:23])
    file.close()

    for j in range(sample_size - sequence_size + (1 + len(missing_imgs))):
        sequence_timestamps = []
        sequence = []

        for e in range(sequence_size):                 
            sequence_timestamps.append(f"{(first_img_dt + dt.timedelta(minutes=time_delta * (j + e))).isoformat()}.000")      # Append all timestamps for the next sequences images to an array in ISO-format. 

        print(sequence_timestamps)
            
        if not any(i in missing_imgs for i in sequence_timestamps):       # Check if all images in sequence_time_arr exist by comparing to missing_imgs
            for i in sequence_timestamps:
                img = load_img(f"{path}/{i.replace(":", "-")}.png")     # Load images as PIL
                img = img.convert("L")      # Converts images to "true gray-scale" (1 channel)
                img_arr = img_to_array(img, dtype=np.uint8)     # converts images to numpy arrays
                print(img_arr.shape, i)
                sequence.append(img_arr)  
            
            sequence = np.stack(sequence, axis=0)     # creates a 4 dimensional numpy array from the python list containing img_arr 
            dataset.append(sequence) 
        else:
            print("Not all images were found in this sequence.")
        
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


# Constructs model and prepares it for training by configuring layers and compiling. 
#
# Paramiters: 
# x_train: Array containg training-data.
#
# Returns:
# model: Model object ready for training.
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


# Load and compile preexisting model to continue training.
#
# Paramiters:
# path: Path to model
# 
# Returns:
#  model: Model object.
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
def save_model(model, path, name, dataset_shape, general_model_info, specific_model_info):
    model.save(f"{path}/{name}.keras", overwrite=False, zipped=True)
    
    with open(f"{path}/{name}_info.txt", "w") as f:
        f.write(f"{general_model_info}Dataset shape: {dataset_shape}\n{specific_model_info}")


#
# PREPARE DATASET
#

# Define modifiable training hyperparameters.
epochs = 20
batch_size = 5

model_name = input("Name of model: ")
general_model_info = f"{input("General info about model: ")}\n\n"
specific_model_info = f"(Epochs: {epochs}\nBatch size: {batch_size}\nFirst layer filters: {input("Number of filters in first layer: ")}\nSecond layer filters: {input("Number of filters in second layer: ")}\nThird layer filters: {input("Number of filters in third layer: ")}\n3D layer filters: {input("Number of filters in 3D layer: ")}\nFirst layer kernel size: {input("Kernel size in first layer: ")}\nSecond layer kernel size: {input("Kernel size in second layer: ")}\nThird layer kernel size: {input("Kernel size in third layer: ")}\n3D layer kernel size: {input("Kernel size in 3D layer: ")}"


dataset = load_dataset("../satellite_imagery_download/images/images", 10, 0, 15)

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
    steps_per_epoch=None,       # default is amount of sequences divided with epochs
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

save_model(model, "../models", model_name, dataset.shape, general_model_info, specific_model_info)