import os
import sys
import warnings
from IPython.core.display import HTML
from IPython.display import Image
from owslib.wcs import WebCoverageService
from owslib.util import Authentication
from owslib.fes import *
from time import sleep
import requests
import xml
from xml.etree import ElementTree
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import eumdac
import ssl

# Turn off SSL certificate verification warnings
ssl._create_default_https_context = ssl._create_unverified_context
warnings.simplefilter("ignore")