for e in range(0, 10):
    print(e)


"""
import numpy as np
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import os

dataset = []
path = "../satellite_imagery_download/images"
for i in range(0, len(os.listdir(path))):   # loops through all images
    img = load_img(f"{path}/{os.listdir(path)[i]}")     # loads images as a PIL image
    img = img.convert("L")      # Converts images to "true gray-scale" (1 channel)
    arr = img_to_array(img)     # converts images to numpy arrays
    print(arr.shape, os.listdir(path)[i])
    dataset.append(arr)     # adds image array to python list
    print("Array added to dataset")

dataset = np.stack(dataset, axis=0)     # creates a 4 dimensional numpy array from the python list of arrays

print(dataset.shape)


# To prove that the code works
img = array_to_img(dataset[0])
img.show()
"""