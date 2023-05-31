"""
Created on maio 31 12:10:55 2023

@author: Ânderson Felipe Weschenfelder
"""
import os
import glob
import cv2 as cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class submatrix:
    def __init__(self, index, Barray, Garray, Rarray, submatriz_length ):
        self.Bmatrix = mount_matrix(Barray, submatriz_length)
        self.Gmatrix = mount_matrix(Garray, submatriz_length)
        self.Rmatrix = mount_matrix(Rarray, submatriz_length)
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

def mount_matrix(array, submatriz_length):
    # global submatriz_length
    matrix = np.empty((submatriz_length), dtype=np.uint8)
    matrix = array
    return matrix

def dividerImage (img, submatriz_height, submatriz_width, submatriz_length, dataframe=None):

    blue = 0  # index da matriz blue da imagem
    green = 1  # index da matriz green da imagem
    red = 2  # index da matriz red da imagem
    listClassSubmatrix = []  # array de classes da submatrizes
    height, width, dim = img.shape
    mBlue, mGreen, mRed = cv2.split(img)  # divide a imagem nas matrizes BGR
    submatriz_num = int((height / submatriz_height) * (width / submatriz_width))
    df = pd.DataFrame()
    # Criar array de matrizes vazio (num, submatrix_height, submatrix_width)
    sumatrix_list = np.empty((submatriz_num, dim) + submatriz_length, dtype=np.uint8)

    # Loop de divisão da matriz
    '''
    Percorre pixel a pixel da matriz, copiando seu valor 
    '''
    m = 0  # index de identificação da submatriz
    # percorre os valores de altura da matriz
    for i in range(0, width, submatriz_width):
        # percorre os valores de largura da matriz
        for j in range(0, height, submatriz_height):
            # Copia para a "m" submatriz o valor do pixel na posição i,j considerando o deslocamento
            # sumatrix_list[m]= img[i:submatriz_width + i, j:submatriz_height + j]
            # obj = submatrix(m, img[i:submatriz_width + i, j:submatriz_height + j].copy())
            sumatrix_list[m, blue] = mBlue[i:submatriz_width + i, j:submatriz_height + j]
            sumatrix_list[m, green] = mGreen[i:submatriz_width + i, j:submatriz_height + j]
            sumatrix_list[m, red] = mRed[i:submatriz_width + i, j:submatriz_height + j]
            obj = submatrix(m, mBlue[i:submatriz_width + i, j:submatriz_height + j],
                            mGreen[i:submatriz_width + i, j:submatriz_height + j],
                            mRed[i:submatriz_width + i, j:submatriz_height + j],
                            submatriz_length)
            listClassSubmatrix.append(obj)
            if dataframe is not None:
                img_df = pd.DataFrame(obj.matrix.reshape(-1)).transpose()
                df = pd.concat([df, img_df], ignore_index=True, axis=0)
            m += 1
    if dataframe is not None:
        return listClassSubmatrix, df

    return listClassSubmatrix