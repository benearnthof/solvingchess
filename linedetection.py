# board recognition try number 2
import cv2
import numpy as np
from os import chdir, getcwd
from matplotlib import pyplot as plt

chdir("C:\\Users\\Bene\\Desktop\\SolvingChess")
getcwd()

img = cv2.imread("screenshot.png")
plt.imshow(img)

# preprocessing 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
cv2.imshow("gray", gray)
cv2.waitKey()
# adaptive threshold
bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

cv2.imshow("gray", bw)
cv2.waitKey()

horizontal = np.copy(bw)
vertical = np.copy(bw)

# Specify size on horizontal axis
cols = horizontal.shape[1]
horizontal_size = cols // 30
# Create structure element for extracting horizontal lines through morphology operations
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
# Apply morphology operations
horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal = cv2.dilate(horizontal, horizontalStructure)

# Specify size on vertical axis
rows = vertical.shape[0]
verticalsize = rows // 30
# Create structure element for extracting vertical lines through morphology operations
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
# Apply morphology operations
vertical = cv2.erode(vertical, verticalStructure)
vertical = cv2.dilate(vertical, verticalStructure)

plt.imshow(vertical)

# Inverse vertical image
vertical = cv2.bitwise_not(vertical)

# Step 1
edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
   
# Step 2
kernel = np.ones((2, 2), np.uint8)
edges = cv2.dilate(edges, kernel)

# Step 3
smooth = np.copy(vertical)
# Step 4
smooth = cv2.blur(smooth, (2, 2))
# Step 5
(rows, cols) = np.where(edges != 0)
vertical[rows, cols] = smooth[rows, cols]

import PIL.Image as PIL_Image
def display_array(array, rng = [0,255]):
    """Plot an array as greyscale image."""
    array = (array - rng[0])/float(rng[1] - rng[0])*255
    array = np.uint8(np.clip(array, 0, 255))
    PIL_Image.fromarray(array).show()
    
# now we do a hough transform to resample into parameter space of lines
total = horizontal + vertical
