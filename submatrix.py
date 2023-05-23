"""
Created on maio 16 19:56:46 2023

@author: Ânderson Felipe Weschenfelder
"""
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

class submatrix:
    def __init__(self, index, array ):
        self.matriz = mount_matrix(array)
        self.index = index
        self.mean = np.mean( self.matriz)
        self.desv = np.std(self.matriz, axis=None)

def mount_matrix(array):
    global submatriz_length

    matriz = np.empty((submatriz_length), dtype=np.uint8)
    matriz = array

    return matriz

img = cv2.imread('Dataset/Testing/fire/abc146.jpg', 0)
# img = cv2.imread('Dataset/Testing/fire/abc146.jpg')

# height, width, dim = img.shape
height, width = img.shape

# Dimensão das submatrizes
'''
A divisão da matriz da imagem pelas submatrizes deve ser um valor inteiro,
caso contrário ocorre erro na divisão da imagem.
EXEMPLO:
height = 250; width = 250
submatriz_height = 125; submatriz_width = 125.

A matriz da imagem original será divida em 4 matrizes menores de tamanho 125x125.
Se fosse submatriz_height = 100 e  submatriz_width = 100, não seria possível devido 
a divisão ser fracionária, resultando em uma matriz incompleta. 
'''
submatriz_height = 25
submatriz_width = 25

# Tamanho das submatrizes
submatriz_length = (submatriz_height, submatriz_width)

# Número de submatrizes
'''
calculo rápido:
submatriz_num = (height/submatrix_height)**2
submatriz_num = (250/25)**2 = 10x10=100
'''
submatriz_num = (height / submatriz_height) * (width / submatriz_width)

if submatriz_num.is_integer():
    submatriz_num = int(submatriz_num)
else:
    print("Número de submatrizes inválido.")
    exit(-1)
# Criar array de matrizes vazio (num, submatrix_height, submatrix_width)
sumatrix_list = np.empty((submatriz_num,) + submatriz_length, dtype=np.uint8)
# sumatrix_list = np.empty((100,3,25,25), dtype=np.uint8)
# variáveis para demarcação de posição da matrix
desl_x = 0
desl_y = 0

lisClass = []

# Loop de divisão da matriz
'''
Percorre pixel a pixel da matriz, copiando seu valor 
'''
# # Cada m repetição é uma nova submatrix
# for m in range(submatriz_num):
#     # percorre os valores de altura da matriz
#     for i in range(submatriz_height):
#         # percorre os valores de largura da matriz
#         for j in range(submatriz_width):
#             # Copia para a "m" submatriz o valor do pixel na posição i,j considerando o deslocamento
#             sumatriz_list[m,i,j]=img[desl_x + i, desl_y + j]
#     # desloca a posição da altura da matrix
#     desl_x += submatriz_height
#     if(desl_x == height):
#         desl_x = 0
#         # desloca a posição de largura da matrix
#         desl_y += submatriz_width
#     if (desl_y == width):
#         desl_y = 0

# Cada m repetição é uma nova submatrix
# for m in range(submatriz_num):
m = 0
    # percorre os valores de altura da matriz
for i in range(0, width,submatriz_width):
    # percorre os valores de largura da matriz
    for j in range(0,height,submatriz_height):
        # Copia para a "m" submatriz o valor do pixel na posição i,j considerando o deslocamento
        sumatrix_list[m]= img[i:submatriz_width + i, j:submatriz_height + j]
        obj = submatrix(m, img[i:submatriz_width + i, j:submatriz_height + j].copy())
        lisClass.append(obj)
        m +=1


'''
Reconstrução da imagem a partir das submatrizes
'''

# # Largura e espessura da borda
# border_h = 2
# border_w = 2
#
# # Cria nova matriz considerando o tamanho da imagem original adicionada da espessura da borda e nº de submatrizes.
# new_image = np.empty((int(height + (height/submatrix_height) * 2 * border_w), int(width + int(width/submatrix_width) * 2 * border_h)),dtype=np.uint8)
#
# # variáveis para demarcação de posição da matrix
# iniciox = 0
# inicioy = 0
# for i in range(submatriz_num):
#     # Adiciona a borda na submatriz
#     previmg = sumatrix_list[i]
#     previmg = cv2.copyMakeBorder(src=previmg, top=border_h, bottom=border_h, left=border_w, right=border_w,
#                                borderType=cv2.BORDER_CONSTANT,value=(255))
#     # Adiciona a submatriz a nova imagem com a borda
#     new_image[iniciox:iniciox + submatrix_height + 2 * border_w , inicioy:inicioy + submatrix_width + 2 * border_h] = previmg
#     # Desloca o inicio da altura da matriz considerando o tamanho de cada submatriz + borda
#     iniciox += submatrix_height + 2 * border_w
#     # Se o inicio da altura chegar ao final da matriz zera ele e desloca o inicio na largura
#     if(iniciox == (height + (height/submatrix_height) * 2 * border_w)):
#         iniciox = 0
#         # Desloca o inicio da largura da matriz considerando o tamanho de cada submatriz + borda
#         inicioy += submatrix_width + 2 * border_h
#     if (inicioy == (width + (width/submatrix_width) * 2 * border_h)):
#         inicioy = 0

# Line thickness of 2 px
thickness = 1

# Blue color in BGR
color = (255, 0, 0)

# represents the top left corner of rectangle
start_point = (0,0)

# Cria nova matriz considerando o tamanho da imagem original adicionada da espessura da borda e nº de submatrizes.
new_image = np.empty((height, width),dtype=np.uint8)

# variáveis para demarcação de posição da matrix
iniciox = 0
inicioy = 0
print("Valores de média e desvio padrão: \r\n")
for i in range(submatriz_num):
    # Adiciona a borda na submatriz
    # previmg = sumatrix_list[i]
    previmg = lisClass[i].matriz
    print("Submatriz [{}] -> Média = {}   Desvio padrão = {}".format(i,lisClass[i].mean, lisClass[i].desv ))
    end_point = (previmg.shape[0], previmg.shape[1])

    # Adicionar o texto na imagem
    texto = "{}".format(i)
    posicao = (0, 20)  # Posição (x, y) do texto na imagem
    fonte = cv2.FONT_HERSHEY_PLAIN
    tamanho_fonte = 1
    cor = (255, 255, 255)  # Cor do texto (no formato BGR)
    cv2.putText(previmg, texto, posicao, fonte, tamanho_fonte, cor, thickness=2)

    # Adicionar um retângulo na imagem
    previmg = cv2.rectangle(previmg, start_point,end_point, color, thickness)

    # Adiciona a submatriz a nova imagem com a borda
    new_image[iniciox:iniciox + submatriz_height, inicioy:inicioy + submatriz_width] = previmg
    # Desloca o inicio da altura da matriz considerando o tamanho de cada submatriz + borda
    inicioy += submatriz_width
    # Se o inicio da altura chegar ao final da matriz zera ele e desloca o inicio na largura
    if inicioy == height:
        inicioy = 0
        # Desloca o inicio da largura da matriz considerando o tamanho de cada submatriz + borda
        iniciox += submatriz_height
    if iniciox == width :
        iniciox = 0

# print(sumatrix_list[0])
# print(img)
cv2.imshow("Imagem Original",img)
cv2.imshow("Imagem de sub Imagens",new_image)
# cv2.imshow("teste",sumatrix_list[1])
cv2.waitKey(0)
cv2.destroyAllWindows()

