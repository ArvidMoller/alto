import numpy as np
import matplotlib.pyplot as plt
import os
import glob
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


def load_model(path):
    name = input("Name of desired model loaded")

    model = keras.saving.load_model(f"{path}/{name}")


def check_perdict_img(path):
    images = os.listdir(f"{__file__[:len(__file__)-11]}{path}")
    
    current_datetime = dt.datetime.now(datetime.timezone.utc)
    current_datetime = (current_datetime + dt.timedelta(minutes=(current_datetime.minute // 15 * 15) - current_datetime.minute - 15)).isoformat()[:16] + ":00.000.png"
    current_datetime = current_datetime.replace(":", "-")

    if len(images) == 0 or images[-1] != current_datetime:
        for i in images:
            os.remove(f"{__file__[:len(__file__)-11]}{path}/{i}")

        download_predict_img(path)
        


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


def download_predict_img(images_path):
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

    start_date = end_date - dt.timedelta(minutes=150)
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







check_perdict_img("/satellite_imagery_download/images/predict_images")

#load dataset

model = load_model("/models")

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