"""
Created on maio 30 21:04:36 2023

@author: Ânderson Felipe Weschenfelder
"""

import csv
import locale
import datetime
import os
import glob
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import submatrix as subM
import Evalution_Test as Ev
import boxplot as bpl

'''
        DIRETÓRIOS
'''
csvfile_training = "training_df.csv"

'''
        DIRETÓRIOS
'''
fireImageDirTraining = 'Dataset/create/Training/*.jpg'
fireImageDirTesting = 'Dataset/create/Testing/*.jpg'
nofireImageDir = 'Dataset/create/nofire/*.jpg'
saveImageDir = 'Dataset/create/save/'

'''
        VARIÁVEIS DE SIMULAÇÃO
'''
blue = 0  # index da matriz blue da imagem
green = 1  # index da matriz green da imagem
red = 2  # index da matriz red da imagem
listClassSubmatrix = []  # array de classes da submatrizes

''' CONFIGURAÇÕES PARA INSERIR TEXTO NA IMAGEM'''
posicao = (0, 20)  # Posição (x, y) do texto na imagem
fonte = cv2.FONT_HERSHEY_PLAIN
tamanho_fonte = 1
cor = (255, 255, 255)  # Cor do texto (no formato BGR)

''' CONFIGURAÇÕES PARA INSERIR RETÂNGULO NA IMAGEM'''
thickness = 1  # Line thickness of 2 px
color = (255, 0, 0)  # Blue color in BGR
start_point = (0, 0)  # represents the top left corner of rectangle

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
csvtestResult = "csvtestResult.csv"
dfresult = pd.DataFrame()
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
if __name__ == "__main__":

    starTime = datetime.datetime.now()
    print("iniciando simulação em: ", starTime.strftime("%H:%M:%S"))
    '''
            TREINAMENTO COM IMAGENS DE TESTE
    '''
    if os.path.exists(csvfile_training):

        teste = pd.DataFrame()
        teste = pd.read_csv(csvtestResult)
        train_df = pd.DataFrame(teste.values)
        train_df = train_df.rename(columns={train_df.columns[-1]: 'Target'})
    else:
        if not os.path.exists(csvtestResult):
            columns = ['time', 'img_train_num', 'img_test_num', 'Features_num', 'tn', 'fp', 'fn', 'tp', 'accuracy', 'TPR', 'FPR', 'Precision', 'F1', 'PLR']
            testeTesting = pd.DataFrame(columns=columns)
        else:
            testeTesting = pd.read_csv(csvtestResult, delimiter=';')

        dfresult['time'] = [starTime.strftime("%H:%M:%S")]
        files_list = glob.glob(fireImageDirTraining)  # imagens contidas no diretório

        imagemcount = 0  # Contador e identificador de imagens
        train_df = pd.DataFrame()  # Dataframe para salvar os dados de cada submatriz
        for files in files_list:
            image_name = os.path.basename(files)
            imagemcount += 1
            img = cv2.imread(files)
            # height, width, dim = img.shape
            # Divide a imagem em 100 submatrizes
            listClassSubmatrix = subM.dividerImage(img, submatriz_height, submatriz_width, submatriz_length)
            # Cria um dataframe identificando cada submatriz com fogo (Target=1) ou sem fogo (Target=o)
            # cada linha representa uma submatrix e contém 25x25x3 colunas, sendo os PIXEL. Ainda é adicionado uma coluna de 'TARGET' para identifcar fogo ou não
            for i in range(len(listClassSubmatrix)):
                img = listClassSubmatrix[i].matrix.copy()  # recebe a matriz do index i

                if image_name == "abc006.jpg":
                    if i == 47 or i == 55 or i == 56 or i == 57 or i == 65 or i == 69 or i == 74 or i == 75 or i == 83 or i == 84 or i == 85 or i == 93 or i == 94:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

                if image_name == "abc008.jpg":
                    if (i >= 10 and i <= 13) or (i >= 20 and i <= 23) or (i >= 30 and i <= 33) or (
                            i >= 40 and i <= 44) or (
                            i >= 52 and i <= 54) or (i >= 63 and i <= 69) or i == 57 or i == 59 or (
                            i >= 73 and i <= 79) or (i >= 83 and i <= 89) or (i >= 98 and i <= 99):
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

                if image_name == "abc040.jpg":
                    if (i >= 71 and i <= 75):
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

                if image_name == "abc057.jpg":
                    if i == 36 or i == 37 or i == 45 or i == 63 or i == 72 or i == 82 or i == 92:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

                if image_name == "abc116.jpg":
                    if i == 9 or (i >= 17 and i <= 19) or (i >= 24 and i <= 29) or (i >= 32 and i <= 39) or (
                            i >= 41 and i <= 49) or (i >= 51 and i <= 59) or (i >= 61 and i <= 65) or (
                            i >= 72 and i <= 75) or (i >= 83 and i <= 84):
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

                if image_name == "fire_0817.jpg":
                    if i == 10 or i == 37 or (i >= 42 and i <= 49) or (i >= 52 and i <= 59) or \
                            (i >= 61 and i <= 67) or (i >= 71 and i <= 72):
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                if image_name == "fire_0697.jpg":
                    if i == 37 or i == 46 or i == 56 or (i >= 60 and i <= 62) or (i >= 66 and i <= 67) or (
                            i >= 71 and i <= 73):
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                elif image_name == "fire_0690.jpg":
                    if (i >= 44 and i <= 46) or (i >= 50 and i <= 52) or (i >= 54 and i <= 56) or (
                            i >= 60 and i <= 62):
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

                elif image_name == "fire_0248.jpg":
                    if i == 9 or i == 19 or i == 48 or i == 57 or i == 76 or i == 82 or i == 91:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                elif image_name == "fire_0205.jpg":
                    if (i >= 20 and i <= 21) or i == 31 or (i >= 42 and i <= 43) or i == 54 or i == 65 or i == 76:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                elif image_name == "fire_0181.jpg":
                    if (i >= 65 and i <= 67) or (i >= 75 and i <= 79) or i == 89:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

                # elif image_name == "fire_0108.jpg":
                #     if i == 18  or i == 28 or (i >= 37 and i <= 38) or (i >= 47 and i <= 48):
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 1
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #     else:
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 0
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                # elif image_name == "fire_0169.jpg":
                #     if (i >= 30 and i <= 33) or (i >= 35 and i <= 36) or (i >= 46 and i <= 47) or i == 57:
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 1
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #     else:
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 0
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                elif image_name == "fire_0194.jpg":
                    if i == 5 or (i >= 12 and i <= 16) or (i >= 22 and i <= 26) or (i >= 32 and i <= 37) or (i >= 42 and i <= 47) \
                            or (i >= 52 and i <= 58) or (i >= 63 and i <= 68) or (i >= 74 and i <= 76) or i == 78:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                elif image_name == "fire_0261.jpg":
                    if i == 4 or i == 35 or (i >= 38 and i <= 39) or (i >= 50 and i <= 51) or (i >= 60 and i <= 61) or i == 68 or \
                            (i >= 70 and i <= 75) or (i >= 80 and i <= 87) or (i >= 90 and i <= 94):
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
             # #############################################################
             #    elif image_name == "fire_0005.jpg":
             #        if i == 32 or (i >= 51 and i <= 52) or i == 56 or i == 61 or (i >= 64 and i <= 66) or (i >= 74 and i <= 76) or \
             #                (i >= 82 and i <= 85) or i == 87 or (i >= 92 and i <= 94):
             #            img_df = subM.mount_Dataframe(listClassSubmatrix[i])
             #            # img_df = pd.DataFrame(img.reshape(-1)).transpose()
             #            img_df['Target'] = 1
             #            train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
             #        else:
             #            img_df = subM.mount_Dataframe(listClassSubmatrix[i])
             #            # img_df = pd.DataFrame(img.reshape(-1)).transpose()
             #            img_df['Target'] = 0
             #            train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
             #    elif image_name == "fire_0085.jpg":
             #        if i == 34 or (i >= 40 and i <= 45) or (i >= 50 and i <= 55) or (i >= 57 and i <= 66) or \
             #                (i >= 73 and i <= 76) or (i >= 80 and i <= 81) or (i >= 90 and i <= 91):
             #            img_df = subM.mount_Dataframe(listClassSubmatrix[i])
             #            # img_df = pd.DataFrame(img.reshape(-1)).transpose()
             #            img_df['Target'] = 1
             #            train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
             #        else:
             #            img_df = subM.mount_Dataframe(listClassSubmatrix[i])
             #            # img_df = pd.DataFrame(img.reshape(-1)).transpose()
             #            img_df['Target'] = 0
             #            train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
             #    elif image_name == "fire_0138.jpg":
             #        if i == 30 or i == 40 or (i >= 50 and i <= 51) or (i >= 53 and i <= 54) or i == 57 or i == 60 or (i >= 63 and i <= 64) or\
             #            (i >= 66 and i <= 67) or i == 69 or (i >= 73 and i <= 77):
             #            img_df = subM.mount_Dataframe(listClassSubmatrix[i])
             #            # img_df = pd.DataFrame(img.reshape(-1)).transpose()
             #            img_df['Target'] = 1
             #            train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
             #        else:
             #            img_df = subM.mount_Dataframe(listClassSubmatrix[i])
             #            # img_df = pd.DataFrame(img.reshape(-1)).transpose()
             #            img_df['Target'] = 0
             #            train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
             #    elif image_name == "fire_0209.jpg":
             #        if i == 13 or i == 33 or (i >= 43 and i <= 44) or (i >= 53 and i <= 55) or (i >= 62 and i <= 67) or \
             #                (i >= 72 and i <= 76) or (i >= 82 and i <= 87):
             #            img_df = subM.mount_Dataframe(listClassSubmatrix[i])
             #            # img_df = pd.DataFrame(img.reshape(-1)).transpose()
             #            img_df['Target'] = 1
             #            train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
             #        else:
             #            img_df = subM.mount_Dataframe(listClassSubmatrix[i])
             #            # img_df = pd.DataFrame(img.reshape(-1)).transpose()
             #            img_df['Target'] = 0
             #            train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
             #    elif image_name == "fire_0262.jpg":
             #        if i == 44 or i == 49 or i == 59 or i == 69:
             #            img_df = subM.mount_Dataframe(listClassSubmatrix[i])
             #            # img_df = pd.DataFrame(img.reshape(-1)).transpose()
             #            img_df['Target'] = 1
             #            train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
             #        else:
             #            img_df = subM.mount_Dataframe(listClassSubmatrix[i])
             #            # img_df = pd.DataFrame(img.reshape(-1)).transpose()
             #            img_df['Target'] = 0
             #            train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
             #    elif image_name == "fire_0308.jpg":
             #        if i == 45 or (i >= 54 and i <= 55) or i == 59 or i == 61 or (i >= 64 and i <= 66) or (i >= 69 and i <= 70) \
             #                or (i >= 72 and i <= 79) or (i >= 82 and i <= 89) or i == 99:
             #            img_df = subM.mount_Dataframe(listClassSubmatrix[i])
             #            # img_df = pd.DataFrame(img.reshape(-1)).transpose()
             #            img_df['Target'] = 1
             #            train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
             #        else:
             #            img_df = subM.mount_Dataframe(listClassSubmatrix[i])
             #            # img_df = pd.DataFrame(img.reshape(-1)).transpose()
             #            img_df['Target'] = 0
             #            train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
             #    elif image_name == "fire_0359.jpg":
             #        if (i >= 57 and i <= 59) or (i >= 67 and i <= 69) or (i >= 76 and i <= 79) or i == 90:
             #            img_df = subM.mount_Dataframe(listClassSubmatrix[i])
             #            # img_df = pd.DataFrame(img.reshape(-1)).transpose()
             #            img_df['Target'] = 1
             #            train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
             #        else:
             #            img_df = subM.mount_Dataframe(listClassSubmatrix[i])
             #            # img_df = pd.DataFrame(img.reshape(-1)).transpose()
             #            img_df['Target'] = 0
             #            train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
             #    elif image_name == "fire_0482.jpg":
             #        if i == 87 or i == 99 or (i >= 80 and i <= 83) or (i >= 89 and i <= 93) or (i >= 96 and i <= 97):
             #            img_df = subM.mount_Dataframe(listClassSubmatrix[i])
             #            # img_df = pd.DataFrame(img.reshape(-1)).transpose()
             #            img_df['Target'] = 1
             #            train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
             #        else:
             #            img_df = subM.mount_Dataframe(listClassSubmatrix[i])
             #            # img_df = pd.DataFrame(img.reshape(-1)).transpose()
             #            img_df['Target'] = 0
             #            train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

                ###################################################

                ###################################################


                elif image_name == "fire_0087.jpg":
                    if i == 67 or (i >= 75 and i <= 76) or (i >= 86 and i <= 87) or i == 89:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                elif image_name == "fire_0107.jpg":
                    if (i >= 8 and i <= 9) or (i >= 17 and i <= 19) or (i >= 28 and i <= 29) or \
                            (i >= 38 and i <= 39) or (i >= 47 and i <= 49) or (i >= 58 and i <= 59) or (i >= 67 and i <= 68):
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                elif image_name == "fire_0120.jpg":
                    if (i >= 15 and i <= 17) or (i >= 25 and i <= 26) or (i >= 35 and i <= 36) or (i >= 45 and i <= 46) or (i >= 55 and i <= 56) or (i >= 64 and i <= 66) \
                        or (i >= 74 and i <= 76) or (i >= 83 and i <= 87) or (i >= 91 and i <= 97):
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                elif image_name == "fire_0167.jpg":
                    if (i >= 40 and i <= 43) or (i >= 47 and i <= 49) or (i >= 56 and i <= 59) \
                            or (i >= 67 and i <= 69) or (i >= 78 and i <= 79) or i == 89:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                elif image_name == "fire_0367.jpg":
                    if (i >= 60 and i <= 61) or (i >= 70 and i <= 71) or (i >= 74 and i <= 75) or (i >= 84 and i <= 85):
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                elif image_name == "fire_0554.jpg":
                    if i == 56 or (i >= 64 and i <= 65) or i == 73 or i == 82:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)


                ##################################################
                ##################################################
                # elif image_name == "fire_0035.jpg":
                #     # if (i >= 40 and i <= 43) or (i >= 47 and i <= 49) or (i >= 56 and i <= 59) \
                #     #         or (i >= 67 and i <= 69) or (i >= 78 and i <= 79) or i == 89:
                #     img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #     # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #     img_df['Target'] = 1
                #     train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #     # else:
                #     #     img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #     #     # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #     #     img_df['Target'] = 0
                #     #     train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                # elif image_name == "fire_0045.jpg":
                #     if (i >= 6 and i <= 61) or i == 63 or  (i >= 65 and i <= 71) or i == 73 or (i >= 75 and i <= 79) \
                #             or i == 82 or i == 86 or i == 93 or (i >= 92 and i <= 93) or (i >= 96 and i <= 99):
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 1
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #     else:
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 0
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                # elif image_name == "fire_0444.jpg":
                #     if(i >= 3 and i <= 9) or (i >= 13 and i <= 19) or (i >= 23 and i <= 29) or (i >= 33 and i <= 39) \
                #         or (i >= 43 and i <= 49) or (i >= 54 and i <= 61) or (i >= 64 and i <= 71) or (i >= 74 and i <= 81) \
                #         or (i >= 85 and i <= 89) or (i >= 96 and i <= 99):
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 1
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #     else:
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 0
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)


                ##################################################
                ##################################################

                elif image_name == "fire_0718.jpg":
                    if (i >= 14 and i <= 15) or (i >= 23 and i <= 24) or (i >= 43 and i <= 44) or \
                            (i >= 52 and i <= 54) or (i >= 62 and i <= 65) or (i >= 72 and i <= 75) \
                            or (i >= 82 and i <= 86) or (i >= 93 and i <= 97):
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)


##########################################################################################################

                # if image_name == "fire_0008.jpg":
                #     if (i >= 35 and i <= 36) or (i >= 43 and i <= 46) or (i >= 54 and i <= 59) or (i >= 64 and i <= 69) or \
                #             (i >= 75 and i <= 79) or (i >= 84 and i <= 87) or (i >= 92 and i <= 97):
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 1
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #     else:
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 0
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #
                # # elif image_name == "fire_0014.jpg":
                # #     if (1 <= i <= 2) or i == 8 or (i >= 11 and i <= 12) or i == 14 or (i >= 16 and i <= 19) or (i >= 21 and i <= 22) \
                # #             or i == 24 or (i >= 26 and i <= 29) or (i >= 31 and i <= 39) or (i >= 41 and i <= 49) or i == 51 \
                # #             or (i >= 57 and i <= 59) or i == 62 or (i >= 65 and i <= 69) or (i >= 71 and i <= 79) \
                # #             or (i >= 81 and i <= 86) or i == 89 or (i >= 93 and i <= 97):
                # #
                # #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                # #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                # #         img_df['Target'] = 1
                # #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                # #     else:
                # #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                # #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                # #         img_df['Target'] = 0
                # #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #
                # elif image_name == "fire_0048.jpg":
                #     if (i >= 6 and i <= 8) or (i >= 17 and i <= 18) or (i >= 26 and i <= 28) or i == 31 or i == 33 or (i >= 35 and i <= 38) or \
                #         i == 41 or (i >= 43 and i <= 48) or i == 51 or (i >= 53 and i <= 58) or (i >= 61 and i <= 68) or (i >= 71 and i <= 74) or \
                #         (i >= 81 and i <= 83) or (i >= 91 and i <= 93):
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 1
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #     else:
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 0
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #
                #
                # elif image_name == "fire_0049.jpg":
                #     if (i >= 3 and i <= 4) or (i >= 12 and i <= 14) or (i >= 22 and i <= 24) or (i >= 32 and i <= 35) or \
                #             (i >= 54 and i <= 56) or (i >= 65 and i <= 66) or i == 69 or (i >= 77 and i <= 79):
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 1
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #     else:
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 0
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #
                #
                # elif image_name == "fire_0051.jpg":
                #     if (i >= 0 and i <= 3) or (i >= 10 and i <= 12) or i == 14 or (i >= 20 and i <= 24) or (
                #             i >= 29 and i <= 34) or (i >= 38 and i <= 44) or (i >= 48 and i <= 54) or (
                #             i >= 58 and i <= 65) or (i >= 69 and i <= 75) or (i >= 78 and i <= 99):
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 1
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #     else:
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 0
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #
                # # elif image_name == "fire_0100.jpg":
                # #     if i == 36 or i == 43 or (i >= 46 and i <= 48) or (i >= 56 and i <= 57) or\
                # #             (i >= 66 and i <= 67) or i == 74 or (i >= 76 and i <= 78) or (i >= 82 and i <= 87) or\
                # #             (i >= 89 and i <= 97) or i == 99:
                # #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                # #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                # #         img_df['Target'] = 1
                # #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                # #     else:
                # #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                # #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                # #         img_df['Target'] = 0
                # #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                # #
                # # elif image_name == "fire_0101.jpg":
                # #     if (i >= 7 and i <= 9) or (i >= 16 and i <= 19) or (i >= 26 and i <= 29) or (
                # #             i >= 35 and i <= 39) or (i >= 44 and i <= 49) or (i >= 54 and i <= 59) or (
                # #             i >= 62 and i <= 69) or (i >= 72 and i <= 79) or (
                # #             i >= 83 and i <= 87) or i == 89 or (i >= 92 and i <= 97):
                # #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                # #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                # #         img_df['Target'] = 1
                # #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                # #     else:
                # #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                # #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                # #         img_df['Target'] = 0
                # #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #
                # elif image_name == "fire_0124.jpg":
                #     if (i >= 39 and i <= 61):
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 1
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #     else:
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 0
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #
                # elif image_name == "fire_0134.jpg":
                #     if i == 23 or (i >= 30 and i <= 33) or (i >= 40 and i <= 42) or (i >= 50 and i <= 54) or (
                #             i >= 60 and i <= 66) or (i >= 70 and i <= 77) or (i >= 80 and i <= 99):
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 1
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #     else:
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 0
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #
                # elif image_name == "fire_0158.jpg":
                #     if i == 24 or (i >= 32 and i <= 34) or i == 36 or (i >= 41 and i <= 44) or (i >= 46 and i <= 49) or (
                #             i >= 51 and i <= 60) or (i >= 63 and i <= 65) or (i >= 70 and i <= 71) or i == 80:
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 1
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #     else:
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 0
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #
                #
                # elif image_name == "fire_0690.jpg":
                #     if (i >= 44 and i <= 46) or (i >= 50 and i <= 52) or (i >= 54 and i <= 56) or (
                #             i >= 60 and i <= 62):
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 1
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #     else:
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 0
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #
                # elif image_name == "fire_0697.jpg":
                #     if i == 37 or i == 46 or i == 56 or (i >= 60 and i <= 62) or (i >= 66 and i <= 67) or (i >= 71 and i <= 73):
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 1
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #     else:
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 0
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #
                # elif image_name == "fire_0817.jpg":
                #     if i == 10 or i == 37 or (i >= 42 and i <= 49) or (i >= 52 and i <= 59) or \
                #             (i >= 61 and i <= 67) or (i >= 71 and i <= 72):
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 1
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                #     else:
                #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                #         img_df['Target'] = 0
                #         train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

        # train_df.to_csv(csvfile_training, index=False)

    # fireboxplot = train_df[train_df['Target'] == 1].copy()
    # nofireboxplot = train_df[train_df['Target'] == 0].copy()


    plt.figure()
    bpl.plotboxplot(train_df.copy())
    # plt.show()

    # SPLITING X and y DATASETS
    y = train_df['Target']
    nofireparts = np.array(train_df['Target'].iloc[list(y == 0)]).flatten()
    fireparts = np.array(train_df['Target'].iloc[list(y == 1)]).flatten()

    print("Fire parts: ", len(fireparts), "No fire parts",len(nofireparts))

    print(y.shape)


    X = train_df
    X.pop('Target')
    print(X.shape)

    # TRAINING KNN
    knn_class = KNeighborsClassifier(n_neighbors=50)
    knn_class.fit(X, y)

    fireImageDir = 'Dataset/Testing/fire/*.jpg'
    nofireImageDir = 'Dataset/Testing/nofire/*.jpg'
    dir = [fireImageDir, nofireImageDir]

    # Ev.ImageTest(knn_class, 'dataset/Testing/fire/abc162.jpg', submatriz_height, submatriz_width, True)
    # Ev.ImageTest(knn_class, 'dataset/Testing/fire/abc080.jpg', submatriz_height, submatriz_width, True)

    # tn, fp, fn, tp = Ev.DirImageTest(knn_class, dir, submatriz_height, submatriz_width)

    # Ev.TestEvaluation(tn, fp, fn, tp)

    tn, fp, fn, tp, test_df = Ev.manualTest(knn_class, fireImageDirTesting, submatriz_height, submatriz_width)

    Ev.TestEvaluation(tn, fp, fn, tp)

    plt.figure()
    bpl.plotboxplot(test_df)
    plt.show()
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    TPR = tp / (tp + fn)
    FPR = fp / (fp + tn)
    Precision = tp / (fp + tp)
    F1 = 2 * (Precision * TPR) / (Precision + TPR)
    PLR = TPR / FPR

    dfresult['img_train_num'] = locale.format_string('%.3f', (X.shape[0]/100))
    dfresult['img_test_num'] = locale.format_string('%.3f', (test_df.shape[0]/100))
    dfresult['Features_num'] = locale.format_string('%.3f', (X.shape[1]))
    dfresult['tn'] = locale.format_string('%.3f', (tn))
    dfresult['fp'] = locale.format_string('%.3f', (fp))
    dfresult['fn'] = locale.format_string('%.3f', (fn))
    dfresult['tp'] = locale.format_string('%.3f', (tp))
    dfresult['accuracy'] = locale.format_string('%.3f', (accuracy * 100))
    dfresult['TPR'] = locale.format_string('%.3f', (TPR * 100))
    dfresult['FPR'] = locale.format_string('%.3f', (FPR * 100))
    dfresult['Precision'] = locale.format_string('%.3f', (Precision * 100))
    dfresult['F1'] = locale.format_string('%.3f', (F1 * 100))
    dfresult['PLR'] = locale.format_string('%.3f', (PLR))

    testeTesting = pd.concat([testeTesting, dfresult], ignore_index=True, axis=0)

    testeTesting.to_csv(csvtestResult, index=False, sep=';')
    stopTime = datetime.datetime.now()
    diftime = stopTime - starTime
    print("Encerando a simulação em: ", stopTime.strftime("%H:%M:%S"))
    print("Duração da simulação: ", diftime.total_seconds())