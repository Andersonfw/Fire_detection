"""
Created on maio 16 20:46:37 2023

@author: Ânderson Felipe Weschenfelder
"""
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * ((out_max - out_min) / (in_max - in_min)) + out_min

img = cv2.imread('imagens/resize_image.jpg', 0)
# img = cv2.imread('Dataset/Testing/fire/abc162.jpg', 0)
print(img.shape)
print(img)
mu = img.shape[0]
tau = img.shape[1]

x_size = img.shape[0]
y_size = img.shape[1]

print(img[1])
print("um pixel",(img[1,1]))

v = np.zeros((mu,tau))
angle = np.zeros((mu,tau))
for x in range(x_size):
    print(x)
    for y in range(y_size):
        for i in range(mu):
            for j in range(tau):
                if i == x or j == y:
                    v[x, y] += 0
                else:
                    value = (1/np.pi) * img[i,j]/((x - i)*(y - j))
                    v[x,y]+= value

# v = 1/np.pi * v

z = img + j*v

ang = np.zeros((mu,tau), dtype=np.uint8)
for i in range(mu):
    for j in range(tau):
        ang[i,j] = math.atan2(v[i,j],img[i,j])

ang=map(ang, np.min(ang), np.max(ang), 0, 255)

print(angle)

plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('image')
plt.subplot(222), plt.imshow(v, cmap='gray'), plt.title('V')
plt.subplot(223), plt.imshow(ang, cmap='gray'), plt.title('angle')

plt.show()

cv2.imshow("image original", img)
cv2.imshow("image sum", v)
cv2.imshow("angle", ang)
cv2.waitKey(0)
cv2.destroyAllWindows()

