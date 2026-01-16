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

diff = cv2.absdiff(img1, img2)

non_zero_pixels = np.count_nonzero(diff)
total_pixels = img1.size

procentage = 100 - ((non_zero_pixels / total_pixels) * 100)

imgs = (img1, img2, diff)
img_names = ("img1", "img2", "diff")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Plot the original frames.
for i, ax in enumerate(axes):
    ax.imshow(imgs[i])
    ax.set_title(img_names[i])
    ax.axis("off")

plt.text(-1300, 600, f"Procentage of matching pixels: {procentage}%", fontsize=22)

# Display the figure.
plt.show()

print(f"Procentage of matching pixels is: {procentage}%")