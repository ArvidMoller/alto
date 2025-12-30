import numpy as np
import os
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import datetime as dt


"""
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
"""



# new "load_dataset" function

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
                img_arr = img_to_array(img)     # converts images to numpy arrays
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


load_dataset("../satellite_imagery_download/images/images", 10, 0, 15)