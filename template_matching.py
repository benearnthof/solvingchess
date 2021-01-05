import PIL.ImageGrab
import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import getcwd as getwd
from os import chdir as setwd

setwd("C:\\Users\\Bene\\Desktop\\SolvingChess")
# im = PIL.ImageGrab.grab()
# im.show()

# image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
# cv2.imwrite("screenshot.png", image)

img_rgb = cv2.imread("screenshot.png")
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('whiteknight.png')
# w, h = template.shape[::-1]
w, h = template.shape[1], template.shape[0]

res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCOEFF_NORMED)
# i think threshold needs to be adjusted 
# the template needs to be RGB so the thresholding works properly
threshold = 0.75
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

# cv2.imwrite('res.png',img_rgb)

# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# todo: wrap everything in functions
# todo: filter coordinates so only one coordinate pair per piece is returned
# todo: translate coordinates into chess position encoding
plt.imshow(img_rgb)