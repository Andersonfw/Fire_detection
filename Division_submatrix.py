"""
Created on maio 16 19:56:46 2023

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
        self.meanall = np.mean( self.matrix)
        self.desvall = np.std(self.matrix, axis=None)
        self.medianall = np.median(self.matrix)
        self.meanBlue = np.mean(self.Bmatrix)
        self.desvBlue = np.std(self.Bmatrix, axis=None)
        self.meanGreen = np.mean(self.Gmatrix)
        self.desvGreen = np.std(self.Gmatrix, axis=None)
        self.meanRed = np.mean(self.Rmatrix)
        self.desvRed = np.std(self.Rmatrix, axis=None)

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
filename = "tabela_medias_1.csv"  # Nome do arquivo CSV
header = ['index', 'mediana','mean global', 'desvio', 'mean blue', 'desv blue', 'mean green', 'desv green', 'mean red',
          'desv red', "R/G mean", "R/B mean", "R/G desv", "R/B desv"]   # Cabeçalho do arquivo CSV
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')  # Configurar a localização para usar a vírgula como separador decimal

'''
        DIRETÓRIOS
'''
fireImageDir = 'Dataset/create/fire/*.jpg"'
nofireImageDir = 'Dataset/create/nofire/*.jpg"'
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
submatriz_height = 10
submatriz_width = 10
submatriz_length = (submatriz_height, submatriz_width)  # Tamanho das submatrizes

'''
        Main
'''
if __name__ == "__main__":
    # img = cv2.imread('Dataset/Testing/fire/abc146.jpg', 0)
    # img = cv2.imread('Dataset/Testing/fire/abc184.jpg')
    # img = cv2.imread('Dataset/Testing/nofire/abc376.jpg')
    img = cv2.imread('Dataset/Testing/fire/abc060.jpg')

    # Apply Gaussian blur to smooth the image (optional)
    # img = cv2.GaussianBlur(src, (5, 5), 1)

    height, width, dim = img.shape
    # height, width = img.shape
    mBlue, mGreen, mRed = cv2.split(img)    # divide a imagem nas matrizes BGR

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
    saveCSVfile(filename,header)    # cria o cabeçalho do arquivo CSV
    # variáveis para demarcação de posição da matrix
    iniciox = 0
    inicioy = 0
    # print("Valores de média e desvio padrão: \r\n")
    for i in range(submatriz_num):
        previmg = listClassSubmatrix[i].matrix.copy() # recebe a matriz do index i
        # print("Submatriz [{}] -> Média = {:.3f}   Desvio padrão = {:.3f}".format(i, listClassSubmatrix[i].mean, listClassSubmatrix[i].desv))

        meanall = locale.format_string('%.3f', listClassSubmatrix[i].meanall)
        desvall = locale.format_string('%.3f', listClassSubmatrix[i].desvall)
        median = locale.format_string('%.3f', listClassSubmatrix[i].medianall)
        meanBlue =  listClassSubmatrix[i].meanBlue
        desvBlue = listClassSubmatrix[i].desvBlue
        meanGreen =  listClassSubmatrix[i].meanGreen
        desvGreen =  listClassSubmatrix[i].desvGreen
        meanRed = listClassSubmatrix[i].meanRed
        desvRed = listClassSubmatrix[i].desvRed

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
            ratio_desv_global = listClassSubmatrix[i].desvall / desvRed

        ratio_mean_RG = locale.format_string('%.3f',ratio_mean_RG)
        ratio_mean_RB = locale.format_string('%.3f', ratio_mean_RB)
        ratio_desv_RG = locale.format_string('%.3f', ratio_desv_RG)
        ratio_desv_RB = locale.format_string('%.3f', ratio_desv_RB)

        meanBlue = locale.format_string('%.3f', np.mean(listClassSubmatrix[i].Bmatrix))
        desvBlue = locale.format_string('%.3f', np.std(listClassSubmatrix[i].Bmatrix, axis=None))
        meanGreen = locale.format_string('%.3f', np.mean(listClassSubmatrix[i].Gmatrix))
        desvGreen = locale.format_string('%.3f', np.std(listClassSubmatrix[i].Gmatrix, axis=None))
        meanRed = locale.format_string('%.3f', np.mean(listClassSubmatrix[i].Rmatrix))
        desvRed = locale.format_string('%.3f', np.std(listClassSubmatrix[i].Rmatrix, axis=None))
        csvdata = [i, median, meanall, desvall, meanBlue, desvBlue, meanGreen, desvGreen,meanRed, desvRed,ratio_mean_RG,ratio_mean_RB,ratio_desv_RG,ratio_desv_RB]
        saveCSVfile(filename, csvdata)

        # Adicionar o texto na imagem
        texto = "{}".format(i)
        cv2.putText(previmg, texto, posicao, fonte, tamanho_fonte, cor, thickness=2)
        # Adicionar um retângulo na imagem
        end_point = (previmg.shape[0], previmg.shape[1])
        previmg = cv2.rectangle(previmg, start_point,end_point, color, thickness)

        # Adiciona a submatriz a nova imagem
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
    '''
    Reconstrução da imagem considerando alguns parâmetros
    '''
    param_image = np.empty((height, width, dim),dtype=np.uint8)
    iniciox = 0
    inicioy = 0
    array_raw = np.zeros((submatriz_height,submatriz_width,dim))    # array nulo para casos que não contem dados relevantes
    for i in range(submatriz_num):

        meanall = listClassSubmatrix[i].meanall
        desvall = listClassSubmatrix[i].desvall
        median = listClassSubmatrix[i].medianall
        meanBlue =  listClassSubmatrix[i].meanBlue
        desvBlue = listClassSubmatrix[i].desvBlue
        meanGreen =  listClassSubmatrix[i].meanGreen
        desvGreen =  listClassSubmatrix[i].desvGreen
        meanRed = listClassSubmatrix[i].meanRed
        desvRed = listClassSubmatrix[i].desvRed

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
            ratio_desv_global = listClassSubmatrix[i].desvall / desvRed


        if desvall > 50 and ratio_desv_global > 0.9 and ratio_mean_RB > 1.1 and ratio_mean_RG > 1.1 and meanRed > 120 and ratio_desv_RB < 5:
        # if desvall > 50 and ratio_desv_global > 1.1 and ratio_mean_RB > 1.1 and ratio_mean_RG > 1.1 and ratio_desv_RB < 5:
            previmg = listClassSubmatrix[i].matrix.copy()
        else:
            previmg = array_raw
        param_image[iniciox:iniciox + submatriz_height, inicioy:inicioy + submatriz_width] = previmg
        inicioy += submatriz_width
        if inicioy == height:
            inicioy = 0
            iniciox += submatriz_height
        if iniciox == width:
            iniciox = 0



    cv2.imshow("Imagem Original", img)
    cv2.imshow("Imagem de sub Imagens", new_image)
    cv2.imshow("Imagem cortada", param_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

