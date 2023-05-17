"""
Created on maio 16 19:56:46 2023

@author: Ânderson Felipe Weschenfelder
"""
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Dataset/Testing/fire/abc168.jpg', 0)

height, width = img.shape

# Dimensão das submatrizes
'''
A divisão da matriz da imagem pelas submatrizes deve ser um valor inteiro,
caso contrário ocorre erro na divisão da imagem.
EXEMPLO:
height = 250; width = 250
submatrix_height = 125; submatrix_width = 125.

A matriz da imagem original será divida em 4 matrizes menores de tamanho 125x125.
Se fosse submatrix_height = 100 e  submatrix_width = 100, não seria possível devido 
a divisão ser fracionária, resultando em uma matriz incompleta. 
'''
submatrix_height = 25
submatrix_width = 25

# Tamanho das submatrizes
submatriz_length = (submatrix_height, submatrix_width)

# Número de submatrizes
'''
calculo rápido:
submatriz_num = (height/submatrix_height)**2
submatriz_num = (250/25)**2 = 10x10=100
'''
submatriz_num = (height/submatrix_height) * (width/submatrix_width)

if submatriz_num.is_integer():
    submatriz_num = int(submatriz_num)
else:
    print("Número de submatrizes inválido.")
    exit(-1)
# Criar array de matrizes vazio (num, submatrix_height, submatrix_width)
sumatrix_list = np.empty((submatriz_num,) + submatriz_length, dtype=np.uint8)

# variáveis para demarcação de posição da matrix
desl_x = 0
desl_y = 0

# Loop de divisão da matriz
'''
Percorre pixel a pixel da matriz, copiando seu valor 
'''
# Cada m repetição é uma nova submatrix
for m in range(submatriz_num):
    # percorre os valores de altura da matriz
    for i in range(submatrix_height):
        # percorre os valores de largura da matriz
        for j in range(submatrix_width):
            # Copia para a "m" submatriz o valor do pixel na posição i,j considerando o deslocamento
            sumatrix_list[m,i,j]=img[desl_x + i,desl_y +j]
    # desloca a posição da altura da matrix
    desl_x += submatrix_height
    if(desl_x == height):
        desl_x = 0
        # desloca a posição de largura da matrix
        desl_y += submatrix_width
    if (desl_y == width):
        desl_y = 0

'''
Reconstrução da imagem a partir das submatrizes
'''

# Largura e espessura da borda
border_h = 2
border_w = 2

# Cria nova matriz considerando o tamanho da imagem original adicionada da espessura da borda e nº de submatrizes.
new_image = np.empty((int(height + (height/submatrix_height) * 2 * border_w), int(width + int(width/submatrix_width) * 2 * border_h)),dtype=np.uint8)

# variáveis para demarcação de posição da matrix
iniciox = 0
inicioy = 0
for i in range(submatriz_num):
    # Adiciona a borda na submatriz
    previmg = sumatrix_list[i]
    previmg = cv2.copyMakeBorder(src=previmg, top=border_h, bottom=border_h, left=border_w, right=border_w,
                               borderType=cv2.BORDER_CONSTANT,value=(255))
    # Adiciona a submatriz a nova imagem com a borda
    new_image[iniciox:iniciox + submatrix_height + 2 * border_w , inicioy:inicioy + submatrix_width + 2 * border_h] = previmg
    # Desloca o inicio da altura da matriz considerando o tamanho de cada submatriz + borda
    iniciox += submatrix_height + 2 * border_w
    # Se o inicio da altura chegar ao final da matriz zera ele e desloca o inicio na largura
    if(iniciox == (height + (height/submatrix_height) * 2 * border_w)):
        iniciox = 0
        # Desloca o inicio da largura da matriz considerando o tamanho de cada submatriz + borda
        inicioy += submatrix_width + 2 * border_h
    if (inicioy == (width + (width/submatrix_width) * 2 * border_h)):
        inicioy = 0

# print(sumatrix_list[0])
# print(img)
cv2.imshow("Imagem Original",img)
cv2.imshow("Imagem de sub Imagens",new_image)
# cv2.imshow("teste",sumatrix_list[1])
cv2.waitKey(0)
cv2.destroyAllWindows()

