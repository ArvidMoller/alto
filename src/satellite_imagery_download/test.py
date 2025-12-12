import cv2
import numpy as np

path = "C:/Users\AI-DatorN\Documents\arvid_olof\alto\src\satellite_imagery_download\images\images\2020-10-01T00-30-00.000.png"

def remove_background(file_name):
    # Read image (BGR format)
    img = cv2.imread(file_name)

    # Define target color (BGR)
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

remove_background(path)