"""
Created on maio 16 19:56:46 2023

@author: Ânderson Felipe Weschenfelder
"""
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Dataset/Testing/fire/abc162.jpg', 0)

# B, G, R = cv2.split(img)

submatrix_x = 125
submatrix_y = 125

B_list = []
G_list = []
R_list = []

# Tamanho das matrizes
tamanho = (submatrix_x, submatrix_y)

# Criar array de matrizes vazio
sumatrix_list = np.empty((4,) + tamanho,dtype=np.uint8)

# sumatrix_list = np.tile(img, (4, 1, 1))
# print((img.shape))
desl_x = 0
desl_y = 0
for m in range(4):
    for i in range(submatrix_x):
        for j in range(submatrix_y):
            sumatrix_list[m,i,j]=img[desl_x + i,desl_y +j]
    desl_x += submatrix_x
    if(desl_x == img.shape[0]):
        desl_x = 0
        desl_y += submatrix_y
    if (desl_y == img.shape[1]):
        desl_y = 0


print(sumatrix_list[0])
print(img)
cv2.imshow("original",img)
cv2.imshow("teste",sumatrix_list[1])
nRows = 25
# Number of columns
mCols = 25

# Dimensions of the image
sizeX = img.shape[1]
sizeY = img.shape[0]

# print(img.shape)
#
# for i in range(0, nRows):
#     for j in range(0, mCols):
#         roi = img[i * sizeY / nRows: i * sizeY / nRows + sizeY / nRows,
#               j * sizeX / mCols:j * sizeX / mCols + sizeX / mCols]
#         cv2.imshow('rois' + str(i) + str(j), roi)
#         # cv2.imwrite('patches/patch_'+str(i)+str(j)+".jpg", roi)
#
#
# # Define the dimensions of the submatrices
# submatrix_height = 25
# submatrix_width = 25

# # Divide the image into submatrices
# submatrices = []
# for y in range(0, img.shape[0], submatrix_height):
#     for x in range(0, img.shape[1], submatrix_width):
#         submatrix = img[y:y + submatrix_height, x:x + submatrix_width]
#         submatrices.append(submatrix)

# Display the submatrices
# for i, submatrix in enumerate(submatrices):
#     cv2.imshow(f'Submatrix {i}', submatrix)
# cv2.imshow(f'Submatrix', submatrix[10])

cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
