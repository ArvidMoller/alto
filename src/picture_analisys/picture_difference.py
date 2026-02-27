import cv2
import numpy as np
import matplotlib.pyplot as plt

compare_type = input("Compare predicted with real? (y/n)").lower()

if compare_type == "y":
    img1 = cv2.imread(f"../../predicted_images/{input("Folder name for predicted img: ")}/{input("Filename: ")}")
    img2 = cv2.imread(f"../satellite_imagery_download/images/images/{input("Filename for real img: ")}")
else:
    img1 = cv2.imread(input("Input full path to img1: "))
    img2 = cv2.imread(input("Input full path to img2: "))

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

within_range_count = 0
for a, b in zip(img1, img2):
    for img1_pixel, img2_pixel in zip(a, b):
        if abs(int(img1_pixel) - int(img2_pixel)) <= 10:
            within_range_count += 1


diff = cv2.absdiff(img1, img2)

total_pixels = img1.size

procentage = (within_range_count / total_pixels) * 100

imgs = (img1, img2, diff)
img_names = ("img1", "img2", "diff")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Plot the original frames.
for i, ax in enumerate(axes):
    ax.imshow(imgs[i])
    ax.set_title(img_names[i])
    ax.axis("off")

plt.text(-1400, 600, f"Procentage of pixels within 10 points: {procentage}%", fontsize=22)

# Display the figure.
plt.show()

print(f"Procentage of pixels within 10 points: {procentage}%")