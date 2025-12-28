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
import argparse
import sys
from pathlib import Path
import time as t

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
target_layer = 'msg_fes__ir108'

# select format option
format_option = 'image/png'

# Define region of interest
region = (-4, 45, 20, 65) # order is lon1,lat1,lon2,lat2

# start time, end time and delta for iteration (year, month, day, hour, minute, second, millisecond)        2020, 10, 1, 00, 00, 00, 000
start_date = datetime.datetime(2020, 10, 10, 00, 00, 00, 000)
end_date = datetime.datetime(2020, 10, 10, 23, 45, 00, 000)
delta = datetime.timedelta(minutes=int(input("Time delta between pictures: (has to be a multiple of 15) ")))


# CURRENTLY NOT IN USE!
#
# Removes background from satellite pictures by making all blue nad green pixles black. 
#
# Parameters:
# file_name: The name of the picture whose background should be removed. 
#
# Returns: 
# void
def remove_background(file_name):
    # Read image (BGR format)
    img = cv2.imread(file_name)

    # Define target color (BGR)
    target_green = np.array([0, 192, 0])  # green in BGR
    target_blue = np.array([255, 0, 0])  # blue in BRG
    tol = 10

    # Create masks for exact matches
    mask_green = np.all(np.abs(img - target_green) < tol, axis=-1)
    mask_blue = np.all(np.abs(img - target_blue) < tol, axis=-1)

    # Combine both masks
    mask = mask_green | mask_blue

    # Set those pixels to black
    img[mask > 0] = [0, 0, 0]

    # Save result
    cv2.imwrite(file_name, img)


#require argument for imagedownload directory
p = argparse.ArgumentParser(
    prog="image_download.py",
    description="Download satellite images")
p.add_argument("images_dir", help="Directory where images will be downloaded") #require extra argument for download directory
#p.add_argument("-v", "--verbose", action="store_true", help="Extra output messages")
p.add_argument("-q", "--quiet", action="store_true", help="Hides (some) output messages")
args = p.parse_args()
images_path = Path(args.images_dir).resolve() #saves argument as a path and removes any extra backslash

#checks if second argument is a valid directory
if not images_path.is_dir():
    print(f"{args.images_dir} is not a valid directory", file=sys.stderr)
    sys.exit(1)
print(f"Downloading images to {images_path}")

# mask_input = input("Should blue and green be changed to black? (y/n)")

start = t.perf_counter()
image_amount = 0

with open("images/download_log.txt", "r+") as f:    # Clear download_log.txt
    f.truncate()


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
    
    for i in range(5):
        try:
            output = wcs.getCoverage(**payload)

            #kod för att spara output bild
            image_filename = f"{time[1].replace(":", "-")}.png"
            with open(images_path / image_filename, "wb") as f: #typ skapar filen, här väljs sökväg och namn, "wb" = writebinary behövs för filer som inte är i textformat (viktigt annars korrupt!)
                f.write(output.read()) #skriver till output med binärkod till PNG filen

            # if mask_input == "y":
            #     remove_background(images_path / image_filename)

            image_amount += 1

            break
        except Exception as err:
            if i < 4:
                print(f"Downlaod of picture at {time[1]} was unsuccessfull. Trying again in 3 seconds.")
                
                t.sleep(3)
            else:
                print(f"Download of of picture at {time[1]} was unsuccessfull 5 times due to error: \n{err}")
                with open("images/download_log.txt", "a") as f:
                    f.write(f"{time[1]}, {image_filename}, Error: {err}\n")



    start_date += delta

    if not args.quiet:
        print(output, time[1])
    
    print(images_path, image_filename)

end = t.perf_counter()
elapsed = end - start

print("")
print(f"Elapsed: {elapsed} seconds, Image Amount: {image_amount}, Avg Time per Image: {elapsed / image_amount} seconds" )