import warnings
from IPython.display import Image
from owslib.wcs import WebCoverageService
from owslib.util import Authentication
from owslib.fes import *
import datetime
import eumdac
import ssl
import datetime

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

# Print the token and its expiration time
# print(f"This token '{token}' expires {token.expiration}")

# Set up the authorization headers for future requests
auth_headers = {"Authorization": f"Bearer {token.access_token}"}

service_url = 'https://view.eumetsat.int/geoserver/wcs?'
wcs = WebCoverageService(service_url, auth=Authentication(verify=False), version='2.0.1', timeout=120)

# Viket layer vi vill ha
target_layer = 'msg_fes__clm'

# select format option
format_option = 'image/png'

# Define region of interest
region = (-20, 35, 25, 65) # order is lon1,lat1,lon2,lat2

# start time, end time and delta for iteration
start_date = datetime.datetime(2020, 10, 1, 10, 00, 00, 000)
end_date = datetime.datetime(2020, 10, 1, 12, 00, 00, 000)
delta = datetime.timedelta(minutes=15)


# iterate over range of dates
while (start_date <= end_date):
    # Set date and time 
    time = [f"{start_date.year}-{start_date.month:02}-{start_date.day:02}T{start_date.hour:02}:{start_date.minute:02}:00.000Z", f"{start_date.year}-{start_date.month:02}-{start_date.day:02}T{start_date.hour:02}:{start_date.minute:02}:00.000"]

    payload = {
        'identifier' : [target_layer,],
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
    with open(f"images/{time[1].replace(":", "-")}.png", "wb") as f: #typ skapar filen, här väljs sökväg och namn, "wb" = writebinary behövs för filer som inte är i textformat (viktigt annars korrupt!)
        f.write(output.read()) #skriver till output med binärkod till PNG filen

    print(output, time[1])