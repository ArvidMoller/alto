import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
import random
import copy
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import SSIM

os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
from keras.preprocessing.image import img_to_array, array_to_img, load_img
from keras import ops

import io
# import imageio
from IPython.display import Image, display
# from ipywidgets import widgets, Layout, HBox

keras.mixed_precision.set_global_policy("mixed_float16")
print("Pytorch version. ", torch.__version__, "GPU name: ", torch.cuda.get_device_name())

ssim_module = SSIM(data_range=2, size_average=True, channel=1, nonnegative_ssim=True)


class ConvLSTMDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float16),
            torch.tensor(self.y[idx], dtype=torch.float16)
        )


# Computes shape of python list.
#
# Parameters:
# arr: Multi-dimensional list.
#
# Returns
# shape: Shape of multi-dimensional list.
def list_shape(arr):
    shape = []
    shape.append(len(arr))
    while True:
        try:
            arr = arr[0]
            shape.append(len(arr))
        except:
            return shape


# Saves all image arrays in an dictionary in order to make sure they are only loaded in into RAM once. 
# 
# Parameters:
# path: Path to images.
# image_cache: Dictionary containing all image arrays associated with their path.
#
# Returns:
# image_cache[path]: The image array associated with the path. 
# image_cache: The dictionary containing all image arrays.
def get_image(path, image_cache, low, high):
    if path not in image_cache:
        img = load_img(path)
        img = img.convert("L")

        # Change based on activation functions, range of input must match activation function. (sigmoid: 0 to 1. relu: 0 to infinty. tanh -1 to 1)
        image_cache[path] = np.float16(
            (img_to_array(img, dtype=np.uint8)) / (255/(high - low)) - abs(low)
        )

    return image_cache[path] , image_cache


# Adds all missing images from download_log.txt to an array
#
# Returns:
# missing_imgs: An array containing the names of all the missing images
def missing_array():
    missing_imgs = []
    file = open("../satellite_imagery_download/images/download_log.txt", "r")       # Open download_log.txt and add all missing timestamps to an array.
    lines = file.readlines()
    for i in lines:
        missing_imgs.append(i.strip()[:23])
    file.close()

    return missing_imgs


# Imports images and converts them from .png to numpy arrays containing 1 channel images. Those images are then added to a sequence array containing the number of images specified in the sequence_size. The sequence array is then appended to a list and converted to a 5-dimensional numpy array. Sequences containing missing images (as specified in ../satellite_imagery_download/images/downlog_log.txt) are skipped.
# 
# Parameters:
# path: Path to images.
# sequence_size: Size of image sequences (time-steps).
# start_index: The index where the download starts.
# time_delta: Time between images (most often 15).
#
# Returns:
# dataset: A 5 dimensional numpy array containing all training images of shape (samples, sequence_size, height, width, channels). 
def load_dataset(path, sequence_size, start_index, time_delta, low, high):
    dataset = []
    image_cache = {}

    total_img_amount = len(os.listdir(path))
    print(f"Total number of pictures: {total_img_amount}")
    sample_size = int(input("Number of training pictures: "))

    while sample_size > total_img_amount:
        print("TOO MANY PICTURES!!!!")
        sample_size = int(input("Number of training pictures: "))

    first_img = os.listdir(path)[start_index][:23]      # Get file name of first image in download.
    
    for i in [13, 16]:      # Convert to ISO-format.
            first_img = first_img[:i] + ":" + first_img[i+1:]

    first_img_dt = dt.datetime.fromisoformat(first_img)     # Convert to datetime format.

    missing_imgs = missing_array()

    for j in range(sample_size - sequence_size + (1 + len(missing_imgs))):
        sequence_timestamps = []
        sequence = []

        for e in range(sequence_size):                 
            sequence_timestamps.append(f"{(first_img_dt + dt.timedelta(minutes=time_delta * (j + e))).isoformat()}.000")      # Append all timestamps for the next sequences images to an array in ISO-format. 

        print(sequence_timestamps)
            
        if not any(i in missing_imgs for i in sequence_timestamps):       # Check if all images in sequence_time_arr exist by comparing to missing_imgs
            for i in tqdm(sequence_timestamps, desc="loading images"):
                img_arr, image_cache = get_image(f"{path}/{i.replace(':','-')}.png", image_cache, low, high)
                print(list_shape(img_arr), i)
                sequence.append(img_arr)  
            
            dataset.append(sequence) 
        else:
            print("Not all images were found in this sequence.")
        
        print("\n")

    print(list_shape(dataset))

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
    train_dataset = []
    val_dataset = []

    indexes = list(range(len(dataset)))
    random.shuffle(indexes)
    train_indexes, val_indexes = indexes[: int((len(indexes)*split_procentage))], indexes[int((len(indexes)*split_procentage)) :]
    
    for i in train_indexes:
        train_dataset.append(dataset[i])

    for i in val_indexes:
        val_dataset.append(dataset[i])

    return train_dataset, val_dataset


# Shifts elements in an array in order to get an 'x' and 'y' array. 
#
# Parameters:
# data: Array containing the training data.
#
# Returns: 
# x: Array containing frames 0 through n-1, where n is the total number of frames in each sequence.
# y: Array containing frames 1 through n, where n is the total number of frames in each sequence.
def shift_frames(data):
    x = copy.deepcopy(data)
    y = copy.deepcopy(data)

    for i in range(len(data)):
        x[i].pop(-1)

    for i in range(len(data)):
        y[i].pop(0)

    return x, y


# def combined_loss(y_true, y_pred):
#     return 0.8 * ops.mean(ops.abs(y_true - y_pred)) + 0.2 * (1 - ssim_module(y_true, y_pred))


def combined_loss(y_true, y_pred):
    # L1 loss
    l1 = ops.mean(ops.abs(y_true - y_pred))

    # Convert to torch tensors explicitly
    y_true_t = torch.as_tensor(y_true)
    y_pred_t = torch.as_tensor(y_pred)

    # y shape: (B, T, H, W, C)
    B, T, H, W, C = y_true_t.shape

    # reshape to (B*T, C, H, W)
    y_true_t = y_true_t.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
    y_pred_t = y_pred_t.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)

    y_true_t = y_true_t.float()
    y_pred_t = y_pred_t.float()

    ssim_loss = 1.0 - ssim_module(y_true_t, y_pred_t)

    return 0.8 * l1 + 0.2 * ssim_loss


# Constructs model and prepares it for training by configuring layers and compiling. 
#
# Paramiters: 
# x_train: Array containg training-data.
#
# Returns:
# model: Model object ready for training.
def construct_model(x_train):
    # Construct the input layer with no definite frame size.
    inp = layers.Input(shape=(None, *(list_shape(x_train)[2:])))

    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = layers.ConvLSTM2D(
        filters=32,                     # few filters
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="tanh",
    )(inp)
    x = layers.LayerNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="tanh",
    )(x)
    x = layers.LayerNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="tanh",
    )(x)
    x = layers.Conv3D(
        filters=1, 
        kernel_size=(3, 3, 3), 
        activation="tanh", 
        padding="same"
    )(x)

    # Next, we will build the complete model and compile it.
    model = keras.models.Model(inp, x)

    model.compile(
        # loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(1e-4),
        loss=combined_loss
    )

    return model


# ¯\_(ツ)_/¯
#
# Parameters:
# dataloader: The dataloader object.
#
# Yeild:
# x: ¯\_(ツ)_/¯
# y: ¯\_(ツ)_/¯
def dataloader_generator(dataloader):
    for x, y in dataloader:
        yield x, y


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
        optimizer=keras.optimizers.Adam(),
        loss=combined_loss
    )

    return model


# Saves the model as .keras file.
#
# Parameters:
# model: The model object.
# path: The path the file should be saved at. 
# name: The name of the saved model.
# ...
# creation_datetime: The date and time when the model began training
# finished_datetime: The date and time when the model finished training
#
# Returns: 
# void
def save_model(model, path, name, dataset_shape, general_model_info, specific_model_info, creation_datetime, finished_datetime, high, low):
    model.save(f"{path}/{name}.keras", overwrite=False, zipped=True)
    
    with open(f"{path}/{name}_info.txt", "w") as f:
        f.write(f"{general_model_info}Dataset shape: {dataset_shape}\nInput range high: {high}\nInput range low: {low}\n{specific_model_info}Model started training {creation_datetime} and finished training {finished_datetime}")


# Finds "model[insert number here].keras" name with lowest number. 
#
# path: Path to directory where model will be saved.
#
# Return:
# name: Name of model. 
def model_name(path):
    model_name_arr = os.listdir(path)
    i = 1
    name = f"model{i}.keras"
    while name in model_name_arr:
        name = f"model{i}.keras"
        i += 1

    return name[:(5+len(str(i)))]


#  ===========================================================================
# PREPARE DATASET
#  ===========================================================================
load_test = input("Is this a loading test? (y/n)").lower()

# Define modifiable training hyperparameters.
epochs = int(input("Number of epochs: "))
batch_size = int(input("Batch size: "))

write_model_info = input("Write information about model? (y/n) ").lower()

if write_model_info == "y":
    name = input("Name of model: ")
    general_model_info = f"{input('General info about model: ')}\n\n"
    specific_model_info = f"Epochs: {epochs}\nBatch size: {batch_size}\nFirst layer filters: {input('Number of filters in first layer: ')}\nSecond layer filters: {input('Number of filters in second layer: ')}\nThird layer filters: {input('Number of filters in third layer: ')}\n3D layer filters: {input('Number of filters in 3D layer: ')}\nFirst layer kernel size: {input('Kernel size in first layer: ')}\nSecond layer kernel size: {input('Kernel size in second layer: ')}\nThird layer kernel size: {input('Kernel size in third layer: ')}\n3D layer kernel size: {input('Kernel size in 3D layer: ')}\n\n"
else:
    name = model_name("../models")
    print(f"Model saved as: {name}.keras")
    general_model_info = "General info about model: Not given\n\n"
    specific_model_info  = f"Epochs: {epochs}\nBatch size: {batch_size}\nFirst layer filters: Not given\nSecond layer filters: \nThird layer filters: Not given\n3D layer filters: Not given\nFirst layer kernel size: Not given\nSecond layer kernel size: Not given\nThird layer kernel size: Not given\n3D layer kernel size: Not given\n\n"

train_on_checkpoint = input("Should training continue on last checkpoint? (y/n) ").lower()

high = int(input("Input range high: "))
low = int(input("Input range low: "))

dataset = load_dataset("../satellite_imagery_download/images/images", 10, 0, 15, low, high)

train_dataset, val_dataset = split_dataset(dataset, 0.9)

# Apply the processing function to the datasets.
x_train, y_train = shift_frames(train_dataset)
x_val, y_val = shift_frames(val_dataset)

# Inspect the dataset.
print("Training Dataset Shapes: " + str(list_shape(x_train)) + ", " + str(list_shape(y_train)))
print("Validation Dataset Shapes: " + str(list_shape(x_val)) + ", " + str(list_shape(y_val)))

if load_test == "y":
    print(x_train[0][0])
    exit()

#  ===========================================================================
#  MODEL TRAINING
#  ===========================================================================

if train_on_checkpoint == "n":
    model = construct_model(x_train)
else:
    model = load_model_for_training("../models/checkpoints")

print("Starting DataLoader")
train_loader = DataLoader(ConvLSTMDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(ConvLSTMDataset(x_val, y_val), batch_size=batch_size, shuffle=True)
print("Pre-processing completed")

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

checkpoint_path = f"../models/checkpoints/{name}_checkpoint.keras"
model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    verbose=0,
    save_best_only=False
    )

creation_datetime = f"{dt.datetime.now()}"

# Fit the model to the training data.
model.fit(
    dataloader_generator(train_loader),
    batch_size=batch_size,
    epochs=epochs,
    steps_per_epoch=len(train_loader),       # default is amount of sequences divided with epochs
    validation_data=dataloader_generator(val_loader),
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

finished_datetime = f"{dt.datetime.now()}"

save_model(model, "../models", name, list_shape(dataset), general_model_info, specific_model_info, creation_datetime, finished_datetime, high, low)