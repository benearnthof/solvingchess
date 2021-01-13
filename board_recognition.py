# board prediction with pytorch
# based on https://github.com/Elucidation/tensorflow_chessbot
import PIL.ImageGrab
import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import chdir, getcwd
import PIL.Image as PIL_Image

# first steps: imageprocessing functions to turn a screenshot into something 
# we can work with
# to show images we can either use plt.imshow or Image.fromarray().show()
chdir("C:\\Users\\Bene\\Desktop\\SolvingChess")

img_file = "screenshot.png"
img = PIL_Image.open(img_file)
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
    PIL_Image.fromarray(array).show()
    
# to locate the board we can use horizontal and vertical kernels 
# opencv should provide us with functions for that but we can also write 
# custom kernels and see which method is fastest
# understanding hough line transform
# cv2.HoughLines returns array of rho and theta values 
# thresholding these values returns the most likely coefficient pairs for lines
# in theta and rho space. converting back to x y space allows to draw the 
# lines that have been detected. 

img = cv2.imread("screenshot.png")
# plt.imshow(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img, 50, 150, apertureSize = 3)
display_array(edges)

lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
display_array(lines)

for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('houghlines3.jpg', img)
# we only get a single line at the top, we need to be more clever to just detect
# the chessboard from the screenshot
# probabilistic hough transform 
img = cv2.imread('screenshot.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
display_array(edges)
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('houghlines5.jpg',img)

# cv2.findchessbaord
def find_chessboard(frame):
    chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    small_frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
    return cv2.findChessboardCorners(small_frame, (7, 7), chessboard_flags)[0] and \
           cv2.findChessboardCorners(frame, (7, 7), chessboard_flags)[0] 

cv2.findChessboardCorners(img, (7,7))
find_chessboard(gray)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img, (x,y), 3, (0,0,255), -1)
    
plt.imshow(img)
display_array(img)

# trying to detect squares as a whole
img = cv2.imread('screenshot.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

thresh = cv2.threshold(sharpen,160,255, cv2.THRESH_BINARY_INV)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

min_area = 100
max_area = 1500
image_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > min_area and area < max_area:
        x,y,w,h = cv2.boundingRect(c)
        ROI = img[y:y+h, x:x+h]
        cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
        cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
        image_number += 1

# cv2.imshow('sharpen', sharpen)
cv2.imshow('close', close)
# cv2.imshow('thresh', thresh)
cv2.imshow('image', img)
cv2.waitKey()

# maybe template matching is gonna work better
from checkerboard import detect_checkerboard

size = (7, 7) # size of checkerboard
image = cv2.imread('screenshot.png') # obtain checkerboard
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(gray, -1, sharpen_kernel)

thresh = cv2.threshold(sharpen,160,255, cv2.THRESH_BINARY_INV)[1]
corners, score = detect_checkerboard(thresh, size)
score

corners_int = corners.astype(int)

for i in corners_int:
    x,y = i.ravel()
    cv2.circle(img, (x,y), 3, (0,0,255), -1)
    
cv2.imshow("image", img)
cv2.waitKey()

# this works but is super slow
