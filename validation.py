"""
Created on maio 24 18:51:50 2023

@author: Ânderson Felipe Weschenfelder
"""

import csv
import locale
import os
import glob
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

class submatrix:
    def __init__(self, index, Barray, Garray, Rarray ):
        self.Bmatrix = mount_matrix(Barray)
        self.Gmatrix = mount_matrix(Garray)
        self.Rmatrix = mount_matrix(Rarray)
        self.matrix = cv2.merge((self.Bmatrix, self.Gmatrix, self.Rmatrix))
        self.index = index
        self.mean = np.mean( self.matrix)
        self.desv = np.std(self.matrix, axis=None)

def mount_matrix(array):
    global submatriz_length
    matrix = np.empty((submatriz_length), dtype=np.uint8)
    matrix = array
    return matrix

def saveCSVfile(filename,data):
    with open(os.path.join(os.getcwd(), filename), 'a+',
              newline='') as f:  # Abre ou cria arquivo csv na pasta documentos do user
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC, delimiter=';')
        writer.writerow(data)

'''
        CSV FILE
'''
filename = "tabela_medias_fire.csv"  # Nome do arquivo CSV
header = ['imagem', 'index', 'mediana','mean global', 'desvio', 'mean blue', 'desv blue', 'mean green', 'desv green', 'mean red',
          'desv red', "R/G mean", "R/B mean", "R/G desv", "R/B desv"]   # Cabeçalho do arquivo CSV
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')  # Configurar a localização para usar a vírgula como separador decimal

'''
        DIRETÓRIOS
'''
fireImageDir = 'Dataset/Testing/fire/*.jpg'
nofireImageDir = 'Dataset/Testing/nofire/*.jpg'
# nofireImageDir = 'Dataset/create/nofire/*.jpg'
saveImageDir = 'Dataset/create/save/'

'''
        VARIÁVEIS DE SIMULAÇÃO
'''
blue = 0    # index da matriz blue da imagem
green = 1    # index da matriz green da imagem
red = 2    # index da matriz red da imagem
listClassSubmatrix = [] # array de classes da submatrizes

''' CONFIGURAÇÕES PARA INSERIR TEXTO NA IMAGEM'''
posicao = (0, 20)  # Posição (x, y) do texto na imagem
fonte = cv2.FONT_HERSHEY_PLAIN
tamanho_fonte = 1
cor = (255, 255, 255)  # Cor do texto (no formato BGR)

''' CONFIGURAÇÕES PARA INSERIR RETÂNGULO NA IMAGEM'''
thickness = 1   # Line thickness of 2 px
color = (255, 0, 0) # Blue color in BGR
start_point = (0, 0)    # represents the top left corner of rectangle

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
submatriz_length = (submatriz_height, submatriz_width)  # Tamanho das submatrizes

'''
        Main
'''
if __name__ == "__main__":

    # saveCSVfile(filename, header)  # cria o cabeçalho do arquivo CSV
    files_list = glob.glob(nofireImageDir)
    imagemcount = 0
    detectcount = 0
    for files in files_list:
        imagemcount += 1
        image_name = os.path.basename(files)
        listClassSubmatrix = []
        img = cv2.imread(files)

        height, width, dim = img.shape
        # height, width = img.shape
        mBlue, mGreen, mRed = cv2.split(img)    # divide a imagem nas matrizes BGR
        submatriz_num = (height / submatriz_height) * (width / submatriz_width)
        if submatriz_num.is_integer():
            submatriz_num = int(submatriz_num)
        else:
            print("Número de submatrizes inválido.")
            exit(-1)

        # Criar array de matrizes vazio (num, submatrix_height, submatrix_width)
        sumatrix_list = np.empty((submatriz_num,dim) + submatriz_length, dtype=np.uint8)

        # variáveis para demarcação de posição da matrix
        desl_x = 0
        desl_y = 0

        # Loop de divisão da matriz
        '''
        Percorre pixel a pixel da matriz, copiando seu valor 
        '''
        m = 0   # index de identificação da submatriz
            # percorre os valores de altura da matriz
        for i in range(0, width,submatriz_width):
            # percorre os valores de largura da matriz
            for j in range(0,height,submatriz_height):
                # Copia para a "m" submatriz o valor do pixel na posição i,j considerando o deslocamento
                # sumatrix_list[m]= img[i:submatriz_width + i, j:submatriz_height + j]
                # obj = submatrix(m, img[i:submatriz_width + i, j:submatriz_height + j].copy())
                sumatrix_list[m, blue]= mBlue[i:submatriz_width + i, j:submatriz_height + j]
                sumatrix_list[m, green] = mGreen[i:submatriz_width + i, j:submatriz_height + j]
                sumatrix_list[m, red] = mRed[i:submatriz_width + i, j:submatriz_height + j]
                obj = submatrix(m, mBlue[i:submatriz_width + i, j:submatriz_height + j],mGreen[i:submatriz_width + i, j:submatriz_height + j],  mRed[i:submatriz_width + i, j:submatriz_height + j])
                listClassSubmatrix.append(obj)
                m += 1
        '''
        Reconstrução da imagem a partir das submatrizes
        '''
        # Cria nova matriz considerando o tamanho da imagem original
        new_image = np.empty((height, width, dim),dtype=np.uint8)

        # variáveis para demarcação de posição da matrix
        iniciox = 0
        inicioy = 0
        for i in range(submatriz_num):
            previmg = listClassSubmatrix[i].matrix.copy() # recebe a matriz do index i

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


        param_image = np.empty((height, width, dim), dtype=np.uint8)
        iniciox = 0
        inicioy = 0
        array_raw = np.zeros(
            (submatriz_height, submatriz_width, dim))  # array nulo para casos que não contem dados relevantes
        detectflag = 0
        for i in range(submatriz_num):
            meanall = listClassSubmatrix[i].mean
            desvall = listClassSubmatrix[i].desv
            median = np.median(listClassSubmatrix[i].matrix)
            meanBlue = np.mean(listClassSubmatrix[i].Bmatrix)
            desvBlue = np.std(listClassSubmatrix[i].Bmatrix, axis=None)
            meanGreen = np.mean(listClassSubmatrix[i].Gmatrix)
            desvGreen = np.std(listClassSubmatrix[i].Gmatrix, axis=None)
            meanRed = np.mean(listClassSubmatrix[i].Rmatrix)
            desvRed = np.std(listClassSubmatrix[i].Rmatrix, axis=None)

            ratio_mean_RG = 0
            ratio_mean_RB = 0
            ratio_desv_RG = 0
            ratio_desv_RB = 10
            ratio_desv_global = 0

            if meanGreen > 0:
                ratio_mean_RG = meanRed / meanGreen
            if meanBlue > 0:
                ratio_mean_RB = meanRed / meanBlue
            if desvGreen > 0:
                ratio_desv_RG = desvRed / desvGreen
            if desvBlue > 0:
                ratio_desv_RB = desvRed / desvBlue

            if desvRed > 0:
                ratio_desv_global = desvall/ desvRed

            # if desvall > 40 and ratio_mean_RB > 1.5 and ratio_mean_RG > 1.5:
            #     previmg = listClassSubmatrix[i].matrix.copy()

            if desvall > 50 and ratio_desv_global > 1.1 and median > 100 and ratio_mean_RB > 1.1 and ratio_mean_RG > 1.1 and meanRed > 120 and ratio_desv_RB < 5:
                detectflag = 1

            inicioy += submatriz_width
            if inicioy == height:
                inicioy = 0
                iniciox += submatriz_height
            if iniciox == width:
                iniciox = 0
        if detectflag:
            detectcount += 1
            print(image_name)

print("De ",imagemcount," foram detectadas ",detectcount," imagens com fogo")

        # caminho_destino = os.path.join(saveImageDir, ("Fireimagem{}_".format(imagemcount) + image_name))
        # cv2.imwrite(caminho_destino, new_image)


