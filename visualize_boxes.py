# This script opens an image and allows the user to click on it to get coordinates for bounding boxes.
# The user should click on the top-left and bottom-right corners of each box.

# NOTE: this script is designed to work on datasets of images with identical layout,
# since the coordinates will be used to crop all images in the dataset equally.
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("5.png")
plt.imshow(img)
plt.title("Click to get box coordinates")
coords = plt.ginput(2*20, timeout=0)  # Click top-left and bottom-right for each, there are 20 boxes
print(coords)

# the printed coordinates can now be used to crop the images
