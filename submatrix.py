"""
Created on maio 16 19:56:46 2023

@author: Ânderson Felipe Weschenfelder
"""
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Dataset/Testing/fire/abc162.jpg', 0)

submatrix_x = 125
submatrix_y = 125

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

teste = sumatrix_list[1]
print(teste.shape)
teste = cv2.copyMakeBorder(src=teste, top=2, bottom=2, left=2, right=2,
                                      borderType=cv2.BORDER_CONSTANT)
cv2.imshow("teste copy ",teste)
print(teste.shape)

border_h = 2
border_w = 2

submatrix_len = sumatrix_list.shape[0]
new_image = np.empty((img.shape[0] + submatrix_len*border_w ,img.shape[1] + submatrix_len*border_h),dtype=np.uint8)

iniciox = 0
inicioy = 0
for i in range(submatrix_len):
    previmg = sumatrix_list[i]
    previmg = cv2.copyMakeBorder(src=previmg, top=border_h, bottom=border_h, left=border_w, right=border_w,
                               borderType=cv2.BORDER_CONSTANT,value=(255))
    new_image[iniciox:iniciox+submatrix_x + 2 * border_w ,inicioy:inicioy+submatrix_y + 2*border_h] = previmg
    iniciox +=submatrix_x + 2 * border_w
    if(iniciox == (img.shape[0] + 4*border_w)):
        iniciox = 0
        inicioy = submatrix_y + 2 * border_h
    if (inicioy == (img.shape[1] + 4*border_h)):
        inicioy = 0

print(sumatrix_list[0])
print(img)
cv2.imshow("original",img)
cv2.imshow("new_Image",new_image)
cv2.imshow("teste",sumatrix_list[1])
cv2.waitKey(0)
cv2.destroyAllWindows()

