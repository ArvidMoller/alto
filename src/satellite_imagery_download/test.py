from keras.preprocessing.image import img_to_array, array_to_img, load_img
import numpy as np

path = "C:/Users/arvid.moller/Desktop/GA/alto/src/satellite_imagery_download/images/images/2025-12-12T11-00-00.000.png"

img = load_img(path)     # loads images as a PIL image
img = img.convert("L")      # Converts images to "true gray-scale" (1 channel)
img_arr = img_to_array(img)
print(img_arr.shape)
img.show()