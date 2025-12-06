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
from IPython.display import Image, display

import warnings
from owslib.wcs import WebCoverageService
from owslib.util import Authentication
from owslib.fes import *
import datetime
import eumdac
import ssl
import datetime
import cv2
import numpy as np


keras.mixed_precision.set_global_policy("mixed_float16")
print("Pytorch version. ", torch.__version__, "GPU name: ", torch.cuda.get_device_name())


def check_perdict_img(path, sequence_size):
    images = os.listdir(f"{__file__[:len(__file__)-11]}{path}")
    
    current_datetime = dt.datetime.now(datetime.timezone.utc)
    current_datetime = (current_datetime + dt.timedelta(minutes=(current_datetime.minute // 15 * 15) - current_datetime.minute - 15)).isoformat()[:16] + ":00.000.png"
    current_datetime = current_datetime.replace(":", "-")

    if len(images) != sequence_size or images[-1] != current_datetime:
        for i in images:
            os.remove(f"{__file__[:len(__file__)-11]}{path}/{i}")

        download_predict_img(path, sequence_size)
        


def remove_background(file_name):
    # Read image (BGR format)
    img = cv2.imread(file_name)

    # Define target color (B, G, R)
    target_green = np.array([0, 192, 0])  # green in BGR
    target_blue = np.array([255, 0, 0])  # blue in BRG

    # Create masks for exact matches
    mask_green = np.all(img == target_green, axis=-1)
    mask_blue = np.all(img == target_blue, axis=-1)

    # Combine both masks
    mask = mask_green | mask_blue

    # Set those pixels to black
    img[mask > 0] = [0, 0, 0]

    # Save result
    cv2.imwrite(file_name, img)


def download_predict_img(images_path, sequence_size):
    # Turn off SSL certificate verification warnings
    ssl._create_default_https_context = ssl._create_unverified_context
    warnings.simplefilter("ignore")

    # Insert your personal key and secret into the single quotes
    consumer_key = 'C7TfIeUyTcIySIZJFfJzQtrSCnga'
    consumer_secret = 'fnflmHejlSooN3clA5vyEnJFVuwa'

    # Provide the credentials (key, secret) for generating a token
    credentials = (consumer_key, consumer_secret)

    # Create a token object from the credentials
    token = eumdac.AccessToken(credentials)

    # Set up the authorization headers for future requests
    auth_headers = {"Authorization": f"Bearer {token.access_token}"}

    service_url = 'https://view.eumetsat.int/geoserver/wcs?'
    wcs = WebCoverageService(service_url, auth=Authentication(verify=False), version='2.0.1', timeout=120)

    # Viket layer vi vill ha
    target_layer = 'msg_fes__clm'

    # select format option
    format_option = 'image/png'

    # Define region of interest
    region = (-4, 45, 20, 65) # order is lon1,lat1,lon2,lat2

    # start time, end time and delta for iteration
    end_date = dt.datetime.now(datetime.timezone.utc)

    end_date = (end_date + dt.timedelta(minutes=(end_date.minute // 15 * 15) - end_date.minute - 15)).isoformat()[:16]
    end_date = dt.datetime.fromisoformat(end_date)

    start_date = end_date - dt.timedelta(minutes=15 * (sequence_size-1))
    delta = datetime.timedelta(minutes=15)

    # iterate over range of dates
    while (start_date <= end_date):
        # Set date and time 
        time = [f"{start_date.year}-{start_date.month:02}-{start_date.day:02}T{start_date.hour:02}:{start_date.minute:02}:00.000Z", f"{start_date.year}-{start_date.month:02}-{start_date.day:02}T{start_date.hour:02}:{start_date.minute:02}:00.000"]

        payload = {
            'identifier' : target_layer,
            'format' : format_option,
            'crs' : 'EPSG:4326',\
            'subsets' : [('Lat',region[1],region[3]),\
                        ('Long',region[0],region[2]), \
                        ('Time',time[0],time[1])],
            'access_token': token
        }
        
        output = wcs.getCoverage(**payload)
        start_date += delta

        #kod för att spara output bild
        image_filename = f"{time[1].replace(':', '-')}.png"
        with open(f"{__file__[:len(__file__)-11]}{images_path}/{image_filename}", "wb") as f: #typ skapar filen, här väljs sökväg och namn, "wb" = writebinary behövs för filer som inte är i textformat (viktigt annars korrupt!)
            f.write(output.read()) #skriver till output med binärkod till PNG filen

        remove_background(f"{__file__[:len(__file__)-11]}{images_path}/{image_filename}")

        print(output, time[1])


def load_dataset(path):
    dataset = []
    samples = os.listdir(f"{__file__[:len(__file__)-11]}{path}")
    print(samples)

    for e in samples:       # loops through the pictures for the next sequence.
        img = load_img(f"{__file__[:len(__file__)-11]}{path}/{e}")     # loads images as a PIL image
        img = img.convert("L")      # Converts images to "true gray-scale" (1 channel)
        img_arr = img_to_array(img)     # converts images to numpy arrays
        print(img_arr.shape, e)
        dataset.append(img_arr)     # adds image array to python list

    dataset = np.stack(dataset, axis=0)     # creates a 5 dimensional numpy array from the python list of arrays

    print(dataset.shape)

    return dataset


def load_model(path):
    name = input("Name of desired model ")

    model = keras.saving.load_model(f"{__file__[:len(__file__)-11]}{path}/{name}.keras")

    print("Model loaded")

    return model


def save_predicted_sequence(predicted_sequence, folder_name):
    current_date = dt.datetime.now(datetime.timezone.utc)
    current_date = (current_date + dt.timedelta(minutes=(current_date.minute // 15 * 15) - current_date.minute - 15)).isoformat()[:16]
    current_date = dt.datetime.fromisoformat(current_date)

    path = f"{__file__[:len(__file__)-14]}{folder_name}/{current_date.isoformat().replace(":", "-")}-00.000"
    print(path)

    try:
        os.mkdir(path)
    except:
        for i in range(1, 1000):
            try:
                path =f"{path}({i})"
                os.mkdir(path)
                break
            except:
                path = path[:(len(path) - 3)]       # remove previous "(number)" ending
    
    
    e = 0
    for i in predicted_sequence:
        img = array_to_img(i)
        img = img.save(f"{path}/{(current_date + dt.timedelta(minutes=15 * (e+1))).isoformat().replace(":", "-")}-00.000.png")
        
        e+=1

    with open(f"{path}/info.txt", "w") as info:
        info.write(f"{input("Info about generated pictures (model settings etc.): ")}")



check_perdict_img("/satellite_imagery_download/images/predict_images", 10)

dataset = load_dataset("/satellite_imagery_download/images/predict_images")

model = load_model("/models")

# Pick the first/last ten frames from the example.
# frames = example[:10, ...]
# original_frames = example[10:, ...]

# Predict a new set of 10 frames.
for i in range(10):
    # Extract the model's prediction and post-process it.
    new_prediction = model.predict(np.expand_dims(dataset, axis=0))
    new_prediction = np.squeeze(new_prediction, axis=0)
    predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

    # Create an array with the predicted frames
    if i == 0:
        predicted_sequence = predicted_frame
    else:
        predicted_sequence = np.append(predicted_sequence, predicted_frame, axis=0)

print("The prediction was successfully made!")

if input("Should predicted images be saved? (y/n) ") == "y":
    save_predicted_sequence(predicted_sequence, "predicted_images")

# Construct a figure for the original and new frames.
fig, axes = plt.subplots(2, 10, figsize=(20, 4))

# Plot the original frames.
for idx, ax in enumerate(axes[0]):
    ax.imshow(np.squeeze(dataset[idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 1}")
    ax.axis("off")

# Plot the new frames.
for idx, ax in enumerate(axes[1]):
    ax.imshow(np.squeeze(predicted_sequence[idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Display the figure.
plt.show()