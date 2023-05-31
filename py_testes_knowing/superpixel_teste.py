"""
Created on maio 12 19:51:45 2023

@author: Ânderson Felipe Weschenfelder
"""

import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import jet
from skimage.color import label2rgb

# Tipo do algoritmo a ser implementado
# ALG = 'SEEDS'
ALG = 'SLIC'
# ALG = 'SLICO'
# ALG = 'MSLIC'
# ALG = 'LSC'


# Open the image colored
src = cv2.imread('imagens/fire_churas.jpg', cv2.IMREAD_COLOR)
assert src is not None, 'Could not open image'

# Resize the image to speed up processing (optional)
src = cv2.resize(src, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# Apply Gaussian blur to smooth the image (optional)
src = cv2.GaussianBlur(src, (3, 3), 0)

# Convert the image to LAB color space
converted = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

# Define the parameters for superpixel segmentation
num_superpixels = 1000  # SEEDS Number of Superpixels
num_levels = 6  # SEEDS Number of Levels
prior = 2  # SEEDS Smoothing Prior
num_histogram_bins = 5  # SEEDS histogram bins
double_step = True  # SEEDS two steps
region_size = 20  # SLIC/SLICO/MSLIC/LSC average superpixel size
ruler = 15.0  # SLIC/MSLIC smoothness (spatial regularization)
ratio = 0.075  # LSC compactness
min_element_size = 25  # SLIC/SLICO/MSLIC/LSC minimum component size percentage
num_iterations = 10  # Iterations

# Create the superpixel object according ALG
if ALG == 'SEEDS':
    superpix = cv2.ximgproc.createSuperpixelSEEDS(
        converted.shape[1], converted.shape[0], converted.shape[2],
        num_superpixels, num_levels, prior, num_histogram_bins, double_step)
elif ALG in {'SLIC', 'SLICO', 'MSLIC'}:
    if ALG == 'SLIC':
        alg  = cv2.ximgproc.SLIC
    elif ALG == 'SLICO':
        alg = cv2.ximgproc.SLICO
    else:
        alg = cv2.ximgproc.MSLIC
    superpix = cv2.ximgproc.createSuperpixelSLIC(
        converted, algorithm=alg, region_size=region_size, ruler=ruler)
elif ALG == 'LSC':
    superpix = cv2.ximgproc.createSuperpixelLSC(
        converted, region_size=region_size, ratio=ratio)
else:
    raise ValueError(f'Unrecognized algorithm {ALG}')

# Perform superpixel segmentation
tic = cv2.getTickCount()
if ALG == 'SEEDS':
    superpix.iterate(converted, num_iterations)
else:
    superpix.iterate(num_iterations)

    # merge small superpixels to neighboring ones
    if min_element_size > 0:
        superpix.enforceLabelConnectivity(min_element_size)

# Get the number of superpixels
npix = superpix.getNumberOfSuperpixels()
print(f'{ALG} segmentation with {npix} superpixels')
toc = (cv2.getTickCount() - tic) / cv2.getTickFrequency()
print(f'Elapsed time: {toc:.6f} s')


# Get the mask for the superpixel boundaries
mask = superpix.getLabelContourMask(thick_line=True)
height,width,channels = converted.shape
color_img = np.zeros((height, width, 3), np.uint8)
color_img[:] = (0, 0, 255)
mask_inv = cv2.bitwise_not(mask)
result_bg = cv2.bitwise_and(src, src, mask=mask_inv)
result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
result = cv2.add(result_bg, result_fg)

# Get the labels for each pixel
labels = superpix.getLabels()
print(labels)
bits = 2
# L = ((labels & (2 ** bits - 1)) * 2 ** (16 - bits)).astype(float) / ((2 ** 16) - 1)
labels = labels.astype(float) + 1
L = label2rgb(labels)# jet(), 'black', 'shuffle')


# Draw the superpixel boundaries on the image
plt.subplot(221), plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)), plt.title('image')
plt.subplot(222), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('superpixel contours')
plt.subplot(223), plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)), plt.title('mask')
plt.subplot(224), plt.imshow(L), plt.title('labels')
plt.show()

# cv2.imshow("Superpixels", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()