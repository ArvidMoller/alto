"""
import numpy as np
import os
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import datetime as dt

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


load_dataset("../satellite_imagery_download/images/images", 10)



# old "load_dataset" function
def load_dataset(path, sequence_size):
    dataset = []
    
    sample_size = len(os.listdir(path))
    print(f"Total number of pictures: {sample_size}")

    sample_size = int(input("Number of training pictures: "))

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

    print(dataset.shape)

    return dataset
"""

with open("test_info.txt", "w") as f:
    f.write(f"{input("General info about model: ")}\nDataset shape: dataset_shape\nEpochs: epochs\nBatch size: batch_size\nFirst layer filters: {input("Number of filters in first layer: ")}\nSecond layer filters: {input("Number of filters in second layer: ")}\nThird layer filters: {input("Number of filters in third layer: ")}\n3D layer filters: {input("Number of filters in 3D layer: ")}\nFirst layer kernel size: {input("Kernel size in first layer: ")}\nSecond layer kernel size: {input("Kernel size in second layer: ")}\nThird layer kernel size: {input("Kernel size in third layer: ")}\n3D layer kernel size: {input("Kernel size in 3D layer: ")}")