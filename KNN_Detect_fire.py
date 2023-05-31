"""
Created on maio 30 21:04:36 2023

@author: Ânderson Felipe Weschenfelder
"""

import csv
import locale
import os
import glob
import cv2 as cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import submatrix as subM

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

def ImageTest(knn, imagename, plot = None):
    global submatriz_height, submatriz_width, submatriz_length
    img_test = cv2.imread(imagename, cv2.IMREAD_COLOR)
    height, width, dim = img_test.shape
    listClassSubmatrixTest, test_df = subM.dividerImage(img_test, submatriz_height, submatriz_width, submatriz_length, True)
    teste = knn.predict(test_df)
    print(teste)
    '''
    Reconstrução da imagem considerando KNN parâmetros
    '''
    param_image = np.empty((height, width, dim),dtype=np.uint8)
    iniciox = 0
    inicioy = 0
    array_raw = np.zeros((submatriz_height,submatriz_width,dim))    # array nulo para casos que não contem dados relevantes
    for i in range(len(teste)):
        if teste[i] == 1:
            previmg = listClassSubmatrixTest[i].matrix.copy()
        else:
            previmg = array_raw
        param_image[iniciox:iniciox + submatriz_height, inicioy:inicioy + submatriz_width] = previmg
        inicioy += submatriz_width
        if inicioy == height:
            inicioy = 0
            iniciox += submatriz_height
        if iniciox == width:
            iniciox = 0
    if plot:
        cv2.imshow("Imagem Original", img_test)
        cv2.imshow("Imagem cortada", param_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return teste



def DirImageTest(knn, dir):
    imagemcount = 0
    detectCorrectcount = 0
    detectIncorrectcount = 0
    for i in range(2):
        files_list = glob.glob(dir[i])
        for files in files_list:
            imagemcount += 1
            image_name = os.path.basename(files)
            img_test = cv2.imread(files)
            listClassSubmatrixTest, test_df = subM.dividerImage(img_test, submatriz_height, submatriz_width, submatriz_length, True)
            teste = knn.predict(test_df)
            if np.max(teste) > 0:
                if i == 0:
                    detectCorrectcount += 1
                else:
                    detectIncorrectcount += 1
            else:
                if i == 0:
                    detectIncorrectcount += 1
                else:
                    detectCorrectcount += 1


    print("De ",imagemcount," foram detectadas ",detectCorrectcount," imagens corretamente")
    print("% de erro/acerto: ",(detectCorrectcount/imagemcount)*100)
    # print("% de erro/acerto: ",100 - (detectcount/imagemcount)*100)


'''
        DIRETÓRIOS
'''
fireImageDir = 'Dataset/create/fire/*.jpg'
nofireImageDir = 'Dataset/create/nofire/*.jpg'
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

    '''
            TREINAMENTO COM IMAGENS DE TESTE
    '''
    files_list = glob.glob(fireImageDir)    # imagens contidas no diretório
    imagemcount = 0 # Contador e identificador de imagens
    train_df = pd.DataFrame()   # Dataframe para salvar os dados de cada submatriz
    for files in files_list:
        imagemcount += 1
        image_name = os.path.basename(files)
        img = cv2.imread(files)
        height, width, dim = img.shape
        # Divide a imagem em 100 submatrizes
        listClassSubmatrix = subM.dividerImage(img, submatriz_height, submatriz_width, submatriz_length)
        # Cria um dataframe identificando cada submatriz com fogo (Target=1) ou sem fogo (Target=o)
        # cada linha representa uma submatrix e contém 25x25x3 colunas, sendo os PIXEL. Ainda é adicionado uma coluna de 'TARGET' para identifcar fogo ou não
        for i in range(len(listClassSubmatrix)):
            img = listClassSubmatrix[i].matrix.copy() # recebe a matriz do index i
            if imagemcount == 1:
                if i == 47 or i == 55 or i == 56 or i == 57 or i == 65 or i == 69 or i == 74 or i == 75 or i == 83 or i == 84 or i == 85 or i == 93 or i == 94:
                    img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

            elif imagemcount == 2:
                if (i >= 10 and i <= 13) or (i >= 20 and i <= 23) or (i >= 30 and i <= 33) or (i >= 40 and i <= 44) or (i >= 52 and i <= 54) or (i >= 63 and i <= 69) or i == 57 or i == 59 or (i >= 73 and i <= 79) or (i >= 83 and i <= 89) or (i >= 98 and i <= 99):
                    img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

            elif imagemcount == 3:
                if (i >= 71 and i <= 75):
                    img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

            elif imagemcount == 4:
                if i == 36 or i == 37 or i == 45 or i == 63 or i == 72 or i == 82 or i == 92:
                    img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

            elif imagemcount == 5:
                if i == 9 or (i >= 17 and i <= 19) or (i >= 24 and i <= 29) or (i >= 32 and i <= 39) or (i >= 41 and i <= 49) or (i >= 51 and i <= 59) or (i >= 61 and i <= 65) or (i >= 72 and i <= 75) or (i >= 83 and i <= 84):
                    img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

    # SPLITING X and y DATASETS
    y = train_df['Target']
    # print(y.shape)

    X = train_df
    X.pop('Target')
    # print(X.shape)

    # TRAINING KNN
    knn_class = KNeighborsClassifier(n_neighbors=50)
    knn_class.fit(X, y)

    fireImageDir = 'Dataset/Training and Validation/fire/*.jpg'
    nofireImageDir = 'Dataset/Training and Validation/nofire/*.jpg'
    dir = [fireImageDir,nofireImageDir]

    ImageTest(knn_class, 'dataset/Testing/fire/abc070.jpg', True)
    # DirImageTest(knn_class,dir)

    # test_df_nofire['Target'] = 0  # TARGET VARIABLE TO FLAG WHEN THERE IS NO FIRE
    #
    # # %%
    # print(test_df_nofire.shape)
    # test_df_nofire.sample(n=15)
    #
    # # %%
    # # CONCATENATING BOTH FIRE AND NON-FIRE DATASETS FOR TESTING
    # test_df = pd.concat([test_df_fire, test_df_nofire], ignore_index=True, axis=0)
    # print(test_df.shape)
    #
    # # %%
    # X_test = test_df.copy()
    # X_test.pop('Target')
    #
    # test_df['Predict'] = knn_class.predict(X_test)
    #
    # # %%
    # # COMPARE TARGET AND PREDICT
    # test_df.loc[test_df['Predict'] == test_df['Target'], 'Error'] = 0
    # test_df.loc[test_df['Predict'] != test_df['Target'], 'Error'] = 1
    #
    # # %%
    # # CONFUSION MATRIX
    # tn, fp, fn, tp = confusion_matrix(test_df['Target'], test_df['Predict']).ravel()
    # (tn, fp, fn, tp)
    #
    # # %%
    # # ACCURACY
    # accuracy = (tp + tn) / len(test_df)
    # print('\nAccuracy: ' + str(accuracy * 100))
    #
    # # TRUE POSITIVE RATE -RECALL
    # TPR = tp / (tp + fn)
    # print('\nTrue Positive Rate(Recall): ' + str(TPR * 100))
    #
    # # FALSE POSITIVE RATE -
    # FPR = fp / (fp + tn)
    # print('False Positive Rate: ' + str(FPR * 100))
    #
    # # PRECISION
    # Precision = tp / (fp + tp)
    # print('Precision Rate: ' + str(Precision * 100))
    #
    # # F1 SCORE
    # F1 = 2 * (Precision * TPR) / (Precision + TPR)
    # print('F1 Score: ' + str(F1 * 100))
    #
    # # POSITIVE LIKELIHOOD RATIO
    # PLR = TPR / FPR
    # print('Positive Likelihood Ratio: ' + str(PLR))