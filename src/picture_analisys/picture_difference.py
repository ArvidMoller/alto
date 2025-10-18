import cv2
import numpy as np

img1 = cv2.imread('../satellite_imagery_download/images/2025-10-17T00-00-00.000.png')
img2 = cv2.imread('../satellite_imagery_download/images/2025-10-18T16-00-00.000.png')

diff = cv2.absdiff(img1, img2)

non_zero_pixels = np.count_nonzero(diff)
total_pixels = img1.size

procetage = 100 - ((non_zero_pixels / total_pixels) * 100)

print(f"Procentage of matching pixels is: {procetage}%")