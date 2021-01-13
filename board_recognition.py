# board prediction with pytorch
# based on https://github.com/Elucidation/tensorflow_chessbot
import PIL.ImageGrab
import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import chdir, getcwd

# first steps: imageprocessing functions to turn a screenshot into something 
# we can work with
# to show images we can either use plt.imshow or Image.fromarray().show()
chdir("C:\\Users\\Bene\\Desktop\\SolvingChess")

img_file = "screenshot.png"
img = Image.open(img_file)
plt.imshow(img)

img.size

# converting to grayscale
grey = np.asarray(img.convert("L"), dtype = np.float32)
plt.imshow(grey)

# writing a custom function that plots in greyscale
def display_array(array, rng = [0,255]):
    """Plot an array as greyscale image."""
    array = (array - rng[0])/float(rng[1] - rng[0])*255
    array = np.uint8(np.clip(array, 0, 255))
    Image.fromarray(array).show()
    
# to locate the board we can use horizontal and vertical kernels 
# opencv should provide us with functions for that but we can also write 
# custom kernels and see which method is fastest
# understanding hough line transform
# cv2.HoughLines returns array of rho and theta values 
# thresholding these values returns the most likely coefficient pairs for lines
# in theta and rho space. converting back to x y space allows to draw the 
# lines that have been detected. 

img = cv2.imread("screenshot.png")
plt.imshow(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(img, 50, 150, apertureSize = 3)
