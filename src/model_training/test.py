import numpy as np
import os
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import datetime as dt


def get_image(path, image_cache):
    if path not in image_cache:
        img = load_img(path)
        img = img.convert("L")
        image_cache[path] = img_to_array(img, dtype=np.uint8)

    return image_cache[path] , image_cache



def load_dataset(path, sequence_size, start_index, time_delta):
    dataset = []
    missing_imgs =[]
    image_cache = {}

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
                img_arr, image_cache = get_image(f"{path}/{i.replace(':','-')}.png", image_cache)
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