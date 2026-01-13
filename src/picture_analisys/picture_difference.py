import cv2
import numpy as np

img1 = cv2.imread("../../predicted_images/2025-12-08T16-30-00-00.000/2025-12-08T16-45-00-00.000.png")
img2 = cv2.imread('../../predicted_images/2025-12-08T16-30-00-00.000/2025-12-08T19-00-00-00.000.png')

diff = cv2.absdiff(img1, img2)

non_zero_pixels = np.count_nonzero(diff)
total_pixels = img1.size

procentage = 100 - ((non_zero_pixels / total_pixels) * 100)

print(f"Procentage of matching pixels is: {procentage}%")