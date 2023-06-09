"""
Created on maio 31 12:10:55 2023

@author: Ânderson Felipe Weschenfelder
"""
import os
import glob
import statistics
from scipy import stats

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
        self.meanall = np.mean(self.matrix)
        self.desvall = np.std(self.matrix, axis=None)
        self.medianall = np.median(self.matrix)
        self.meanBlue = np.mean(self.Bmatrix)
        self.desvBlue = np.std(self.Bmatrix, axis=None)
        self.meanGreen = np.mean(self.Gmatrix)
        self.desvGreen = np.std(self.Gmatrix, axis=None)
        self.meanRed = np.mean(self.Rmatrix)
        self.desvRed = np.std(self.Rmatrix, axis=None)

def mount_matrix(array, submatriz_length):
    matrix = np.empty((submatriz_length), dtype=np.uint8)
    matrix = array
    return matrix

def mount_Dataframe(submatrixClass):
    df = pd.DataFrame()
    # df = pd.DataFrame(submatrixClass.matrix.copy().reshape(-1)).transpose()
    # dfb = pd.DataFrame(submatrixClass.Bmatrix.copy().reshape(-1)).transpose()
    # dfg = pd.DataFrame(submatrixClass.Gmatrix.copy().reshape(-1)).transpose()
    # dfr = pd.DataFrame(submatrixClass.Rmatrix.copy().reshape(-1)).transpose()
    # df = pd.concat([df, dfb], axis=1)
    # df = pd.concat([df, dfg], axis=1)
    # df = pd.concat([df, dfr], axis=1)

    #########################   ALL FEATURES  #######################

    # df[0] = [submatrixClass.desvRed]
    # df[df.columns[-1] + 1] = submatrixClass.desvGreen
    # df[df.columns[-1] + 1] = submatrixClass.desvBlue
    # df[df.columns[-1] + 1] = submatrixClass.desvall
    # # df[df.columns[-1] + 1] = (submatrixClass.desvBlue + submatrixClass.desvGreen + submatrixClass.desvall) / 3
    # df[df.columns[-1] + 1] = submatrixClass.meanRed
    # df[df.columns[-1] + 1] = submatrixClass.meanBlue
    # df[df.columns[-1] + 1] = submatrixClass.meanGreen
    # df[df.columns[-1] + 1] = (submatrixClass.meanRed + submatrixClass.meanGreen + submatrixClass.meanBlue) / 3
    # df[df.columns[-1] + 1] = submatrixClass.medianall
    # df[df.columns[-1] + 1] = np.median(submatrixClass.Bmatrix)
    # df[df.columns[-1] + 1] = np.median(submatrixClass.Gmatrix)
    # df[df.columns[-1] + 1] = np.median(submatrixClass.Rmatrix)
    # df[df.columns[-1] + 1] = np.argmax(np.bincount(submatrixClass.Rmatrix.flatten()))
    # df[df.columns[-1] + 1] = np.argmax(np.bincount(submatrixClass.Gmatrix.flatten()))
    # df[df.columns[-1] + 1] = np.argmax(np.bincount(submatrixClass.Bmatrix.flatten()))
    # df[df.columns[-1] + 1] = stats.hmean(submatrixClass.Gmatrix, axis=None)
    # df[df.columns[-1] + 1] = stats.hmean(submatrixClass.Rmatrix, axis=None)
    # df[df.columns[-1] + 1] = stats.hmean(submatrixClass.Bmatrix, axis=None)
    # df[df.columns[-1] + 1] = np.min(submatrixClass.Rmatrix)
    # df[df.columns[-1] + 1] = entropyCalcGrayScale(submatrixClass.Rmatrix)
    # df[df.columns[-1] + 1] = entropyCalcGrayScale(submatrixClass.Gmatrix)
    # df[df.columns[-1] + 1] = entropyCalcGrayScale(submatrixClass.Bmatrix)
    #################################################################

    ########################  TESTES  ##########################
    # # df[df.columns[-1] + 1] = np.std(padronizar(submatrixClass.Bmatrix.copy())[0],  axis=None)
    # # df[df.columns[-1] + 1] = np.std(padronizar(submatrixClass.Gmatrix.copy())[0],  axis=None)
    # # df[df.columns[-1] + 1] = np.std(padronizar(submatrixClass.matrix.copy())[0],  axis=None)
    # # df[df.columns[-1] + 1] = np.mean(padronizar(submatrixClass.Rmatrix.copy())[0])
    # # df[df.columns[-1] + 1] = np.mean(padronizar(submatrixClass.Gmatrix.copy())[0])
    # # df[df.columns[-1] + 1] = np.mean(padronizar(submatrixClass.matrix.copy())[0])
    # # df[df.columns[-1] + 1] = np.median(padronizar(submatrixClass.matrix.copy())[0])

    img = cv2.cvtColor(submatrixClass.matrix, cv2.COLOR_BGR2YCrCb)
    Y, Cb, Cr = cv2.split(img)
    mean_y = np.mean(Y)
    mean_cb = np.mean(Cb)
    mean_cr = np.mean(Cr)
    # df[df.columns[-1] + 1] = mean_y
    # df[df.columns[-1] + 1] = mean_cb
    # # df[df.columns[-1] + 1] = mean_cr

    #################################################################
    df[0] = [submatrixClass.desvall]
    # df[df.columns[-1] + 1] = submatrixClass.desvGreen
    # df[df.columns[-1] + 1] = submatrixClass.desvBlue
    # df[df.columns[-1] + 1] = submatrixClass.desvall
    # df[df.columns[-1] + 1] = (submatrixClass.desvBlue + submatrixClass.desvGreen + submatrixClass.desvall) / 3
    # df[df.columns[-1] + 1] = np.std(np.concatenate((submatrixClass.Rmatrix.flatten(), submatrixClass.Gmatrix.flatten(),submatrixClass.Bmatrix.flatten())),  axis=None)
    df[df.columns[-1] + 1] = submatrixClass.meanRed
    # df[df.columns[-1] + 1] = submatrixClass.meanBlue
    # df[df.columns[-1] + 1] = submatrixClass.meanGreen
    # df[df.columns[-1] + 1] = submatrixClass.meanall
    # df[df.columns[-1] + 1] = (submatrixClass.meanRed + submatrixClass.meanGreen + submatrixClass.meanBlue) / 3
    # df[df.columns[-1] + 1] = submatrixClass.medianall
    # df[df.columns[-1] + 1] = np.median(submatrixClass.Bmatrix)
    # df[df.columns[-1] + 1] = np.median(submatrixClass.Gmatrix)
    df[df.columns[-1] + 1] = np.median(submatrixClass.Rmatrix)
    df[df.columns[-1] + 1] = np.argmax(np.bincount(submatrixClass.Rmatrix.flatten()))
    # df[df.columns[-1] + 1] = np.argmax(np.bincount(submatrixClass.Gmatrix.flatten()))
    # df[df.columns[-1] + 1] = np.argmax(np.bincount(submatrixClass.Bmatrix.flatten()))
    # df[df.columns[-1] + 1] = stats.hmean(submatrixClass.Gmatrix, axis=None)
    df[df.columns[-1] + 1] = stats.hmean(submatrixClass.Rmatrix, axis=None)
    # df[df.columns[-1] + 1] = stats.hmean(submatrixClass.Bmatrix, axis=None)
    # df[df.columns[-1] + 1] = np.min(submatrixClass.Rmatrix)
    # df[df.columns[-1] + 1] = entropyCalcGrayScale(submatrixClass.Rmatrix)
    # df[df.columns[-1] + 1] = entropyCalcGrayScale(submatrixClass.Gmatrix)
    # df[df.columns[-1] + 1] = entropyCalcGrayScale(submatrixClass.Bmatrix)
    # df[0] = [submatrixClass.desvall]
    # df[df.columns[-1] + 1] = mean_cb
    # df[df.columns[-1] + 1] = np.median(submatrixClass.Rmatrix)
    # df[df.columns[-1] + 1] = np.argmax(np.bincount(submatrixClass.Rmatrix.flatten()))
    # df[df.columns[-1] + 1] = stats.hmean(submatrixClass.Rmatrix, axis=None)

    ####################### BEST OPTION  #####################
    # df[0] = [submatrixClass.desvall]
    # df[df.columns[-1] + 1] = submatrixClass.meanRed
    # df[df.columns[-1] + 1] = np.median(submatrixClass.Rmatrix)
    # df[df.columns[-1] + 1] = np.argmax(np.bincount(submatrixClass.Rmatrix.flatten()))
    # df[df.columns[-1] + 1] = stats.hmean(submatrixClass.Rmatrix, axis=None)
    return df

def padronizar(entrada):
    if entrada.std() == 0:
        return 0,0,0
    return (entrada - entrada.mean())/(entrada.std()), entrada.mean(),entrada.std()
def entropyCalcGrayScale(imagem):
    entropy = 0
    tam_imagem = imagem.shape
    total = tam_imagem[0] * tam_imagem[1]
    hist = cv2.calcHist([imagem], [0], None, [256], [0, 256])
    for i in range(len(hist)):
        if (hist[i] > 0):
            temp = (hist[i] / total) * np.log2(hist[i] / total)
            entropy = entropy + temp

    entropy = entropy[0] * -1
    return entropy
def dividerImage (img, submatriz_height, submatriz_width, submatriz_length, dataframe=None):

    blue = 0  # index da matriz blue da imagem
    green = 1  # index da matriz green da imagem
    red = 2  # index da matriz red da imagem
    listClassSubmatrix = []  # array de classes da submatrizes
    width, height, dim = img.shape
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
                img_df = mount_Dataframe(obj)
                # img_df = pd.DataFrame(obj.matrix.reshape(-1)).transpose()
                df = pd.concat([df, img_df], ignore_index=True, axis=0)
            m += 1
    if dataframe is not None:
        return listClassSubmatrix, df

    return listClassSubmatrix