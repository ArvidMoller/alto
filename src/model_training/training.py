#
#  NYTT NAMN BEHÖVS!
#

import numpy as np
import matplotlib.pyplot as plt
import os

import torch

os.environ["KERAS_BACKEND"] = "torch"
print("Pytorch version. ", torch.__version__, "GPU name: ", torch.cuda.get_device_name())

import keras
from keras import layers
from keras.preprocessing.image import img_to_array, array_to_img, load_img

import io
# import imageio
from IPython.display import Image, display
# from ipywidgets import widgets, Layout, HBox

# Imports images and converts them from .png to numpy arrays containing 1 channel images. Those images are then added to a sequence array containing the number of images specified in the sequence_size. The sequence array is then appended to a list and converted to a 5-dimensional numpy array. 
# 
# Parameters:
# path: Path to images.
# sequence_size: Size of image sequences (time-steps).
#
# Returns:
# dataset: A 5 dimensional numpy array containing all training images of shape (samples, sequence_size, height, width, channels). 
def load_dataset(path, sequence_size):
    dataset = []
    sample_size = len(os.listdir(path))
    print(sample_size)
    for i in range(0, sample_size - (sequence_size-1)):   # loops through all images up to and including the image sequence_size places from the last.
        sequence = []

        for e in range(0, sequence_size):       # loops through the pictures for the next sequence.
            img = load_img(f"{path}/{os.listdir(path)[i + e]}")     # loads images as a PIL image
            img = img.convert("L")      # Converts images to "true gray-scale" (1 channel)
            img_arr = img_to_array(img)     # converts images to numpy arrays
            print(img_arr.shape, os.listdir(path)[i+e], i+e)
            sequence.append(img_arr)     # adds image array to python list
    
        sample = np.stack(sequence, axis=0)     # creates a 4 dimensional numpy array from the python list containing img_arr 
        dataset.append(sample)      # adds sample array to python list

    dataset = np.stack(dataset, axis=0)     # creates a 5 dimensional numpy array from the python list of arrays

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



dataset = load_dataset("../satellite_imagery_download/images/images", 10)
# print(dataset.shape)

train_dataset, val_dataset = split_dataset(dataset, 0.9)






# Normalize the data to the 0-1 range.
train_dataset = train_dataset / 255
val_dataset = val_dataset / 255


# We'll define a helper function to shift the frames, where
# `x` is frames 0 to n - 1, and `y` is frames 1 to n.
def create_shifted_frames(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    return x, y


# Apply the processing function to the datasets.
x_train, y_train = create_shifted_frames(train_dataset)
x_val, y_val = create_shifted_frames(val_dataset)

# Inspect the dataset.
print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

#
#  DATA VISUALISERING
#

# # Construct a figure on which we will visualize the images.
# fig, axes = plt.subplots(4, 5, figsize=(10, 8))

# # Plot each of the sequential images for one random data example.
# data_choice = np.random.choice(range(len(train_dataset)), size=1)[0]
# for idx, ax in enumerate(axes.flat):
#     ax.imshow(np.squeeze(train_dataset[data_choice][idx]), cmap="gray")
#     ax.set_title(f"Frame {idx + 1}")
#     ax.axis("off")

# # Print information and display the figure.
# print(f"Displaying frames for example {data_choice}.")
# plt.show()

#
#  MODEL KONSTRUKTION (stavfel?) (kasnek, jag kan inte stava, // möller) kanel
#

# Construct the input layer with no definite frame size.
inp = layers.Input(shape=(None, *x_train.shape[2:]))

# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
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

#
#  MODEL TRÄNING
#

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# Define modifiable training hyperparameters.
epochs = 20
batch_size = 5

# Fit the model to the training data.
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr],
)

#
#  BILD FÖRUTSÄGELSE VISUALISERING
#

# Select a random example from the validation dataset.
example = val_dataset[np.random.choice(range(len(val_dataset)), size=1)[0]]

# Pick the first/last ten frames from the example.
frames = example[:10, ...]
original_frames = example[10:, ...]

# Predict a new set of 10 frames.
for _ in range(10):
    # Extract the model's prediction and post-process it.
    new_prediction = model.predict(np.expand_dims(frames, axis=0))
    new_prediction = np.squeeze(new_prediction, axis=0)
    predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

    # Extend the set of prediction frames.
    frames = np.concatenate((frames, predicted_frame), axis=0)

# Construct a figure for the original and new frames.
fig, axes = plt.subplots(2, 10, figsize=(20, 4))

# Plot the original frames.
for idx, ax in enumerate(axes[0]):
    ax.imshow(np.squeeze(original_frames[idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Plot the new frames.
new_frames = frames[10:, ...]
for idx, ax in enumerate(axes[1]):
    ax.imshow(np.squeeze(new_frames[idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Display the figure.
plt.show()