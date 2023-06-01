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
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import submatrix as subM

def ImageTest(knn, imagename, plot=None):
    global submatriz_height, submatriz_width, submatriz_length
    img_test = cv2.imread(imagename, cv2.IMREAD_COLOR)
    height, width, dim = img_test.shape
    listClassSubmatrixTest, test_df = subM.dividerImage(img_test, submatriz_height, submatriz_width, submatriz_length,
                                                        True)
    teste = knn.predict(test_df)
    print("Predict Result: ", teste)
    '''
    Reconstrução da imagem considerando KNN parâmetros
    '''
    param_image = np.empty((height, width, dim), dtype=np.uint8)
    iniciox = 0
    inicioy = 0
    array_raw = np.zeros(
        (submatriz_height, submatriz_width, dim))  # array nulo para casos que não contem dados relevantes
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


def manualTest(knn, dir):
    global submatriz_height, submatriz_width, submatriz_length
    imagemcount = 0
    files_list = glob.glob(dir)
    test_df = pd.DataFrame()  # Dataframe para salvar os dados de cada submatriz
    for files in files_list:
        imagemcount += 1
        img_test = cv2.imread(files)
        # Divide a imagem em 100 submatrizes
        listClassSubmatrix = subM.dividerImage(img_test, submatriz_height, submatriz_width, submatriz_length)
        # Cria um dataframe identificando cada submatriz com fogo (Target=1) ou sem fogo (Target=o)
        # cada linha representa uma submatrix e contém 25x25x3 colunas, sendo os PIXEL. Ainda é adicionado uma coluna de 'TARGET' para identifcar fogo ou não
        for i in range(len(listClassSubmatrix)):
            img = listClassSubmatrix[i].matrix.copy()  # recebe a matriz do index i
            if imagemcount == 1:
                if i == 47 or i == 55 or i == 56 or i == 57 or i == 65 or i == 69 or i == 74 or i == 75 or i == 83 or i == 84 or i == 85 or i == 93 or i == 94:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)

            elif imagemcount == 2:
                if (i >= 10 and i <= 13) or (i >= 20 and i <= 23) or (i >= 30 and i <= 33) or (i >= 40 and i <= 44) or (
                        i >= 52 and i <= 54) or (i >= 63 and i <= 69) or i == 57 or i == 59 or (
                        i >= 73 and i <= 79) or (i >= 83 and i <= 89) or (i >= 98 and i <= 99):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)

            elif imagemcount == 3:
                if (i >= 71 and i <= 75):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)

            elif imagemcount == 4:
                if i == 36 or i == 37 or i == 45 or i == 63 or i == 72 or i == 82 or i == 92:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)

            elif imagemcount == 5:
                if i == 9 or (i >= 17 and i <= 19) or (i >= 24 and i <= 29) or (i >= 32 and i <= 39) or (
                        i >= 41 and i <= 49) or (i >= 51 and i <= 59) or (i >= 61 and i <= 65) or (
                        i >= 72 and i <= 75) or (i >= 83 and i <= 84):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
    return test_df

# tn = true negative ( with nofire detect correctly)
# fp = false positive ( with nofire detect incorrectly)
# fn = false negative ( with nofire detect incorrectly)
# tp = true positive ( with fire detect correctly)
def DirImageTest(knn, dir):
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    imagemcount = 0
    detectCorrectcount = 0
    detectIncorrectcount = 0
    for i in range(2):
        files_list = glob.glob(dir[i])
        for files in files_list:
            imagemcount += 1
            image_name = os.path.basename(files)
            img_test = cv2.imread(files)
            listClassSubmatrixTest, test_df = subM.dividerImage(img_test, submatriz_height, submatriz_width,
                                                                submatriz_length, True)
            teste = knn.predict(test_df)
            if np.max(teste) > 0:
                if i == 0:
                    detectCorrectcount += 1
                    tp += 1
                else:
                    detectIncorrectcount += 1
                    fp += 1
            else:
                if i == 0:
                    detectIncorrectcount += 1
                    fn += 1
                else:
                    detectCorrectcount += 1
                    tn += 1

    print("De ", imagemcount, " foram detectadas ", detectCorrectcount, " imagens corretamente")
    print("% de acerto: ", (detectCorrectcount / imagemcount) * 100)
    return tn, fp, fn, tp

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
if __name__ == "__main__":

    starTime = datetime.datetime.now()
    print("iniciando simulação em: ", starTime.strftime("%H:%M:%S"))
    '''
            TREINAMENTO COM IMAGENS DE TESTE
    '''
    if os.path.exists(csvfile_training):
        train_df = pd.read_csv(csvfile_training)
    else:
        files_list = glob.glob(fireImageDirTraining)  # imagens contidas no diretório
        imagemcount = 0  # Contador e identificador de imagens
        train_df = pd.DataFrame()  # Dataframe para salvar os dados de cada submatriz
        for files in files_list:
            imagemcount += 1
            img = cv2.imread(files)
            # height, width, dim = img.shape
            # Divide a imagem em 100 submatrizes
            listClassSubmatrix = subM.dividerImage(img, submatriz_height, submatriz_width, submatriz_length)
            # Cria um dataframe identificando cada submatriz com fogo (Target=1) ou sem fogo (Target=o)
            # cada linha representa uma submatrix e contém 25x25x3 colunas, sendo os PIXEL. Ainda é adicionado uma coluna de 'TARGET' para identifcar fogo ou não
            for i in range(len(listClassSubmatrix)):
                img = listClassSubmatrix[i].matrix.copy()  # recebe a matriz do index i

                if imagemcount == 11:
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

                elif imagemcount == 12:
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

                elif imagemcount == 13:
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

                elif imagemcount == 14:
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

                elif imagemcount == 15:
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



                if imagemcount == 1:
                    if (i >= 35 and i <= 36) or (i >= 43 and i <= 46) or (i >= 53 and i <= 58) or (i >= 65 and i <= 69) or (i >= 75 and i <= 79) or (i >= 84 and i <= 87) or (i >= 92 and i <= 97):
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

                elif imagemcount == 2:
                    if (1 <= i <= 2) or i == 8 or (i >= 11 and i <= 12) or i == 14 or (i >= 16 and i <= 19) or (i >= 21 and i <= 22) \
                            or i == 24 or (i >= 26 and i <= 29) or (i >= 31 and i <= 39) or (i >= 41 and i <= 49) or i == 51 \
                            or (i >= 57 and i <= 59) or i == 62 or (i >= 65 and i <= 69) or (i >= 71 and i <= 79) \
                            or (i >= 81 and i <= 86) or i == 89 or (i >= 93 and i <= 97):

                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

                elif imagemcount == 3:
                    if (i >= 7 and i <= 8) or (i >= 17 and i <= 18) or (i >= 27 and i <= 28) or (i >= 36 and i <= 38) or \
                        (i >= 43 and i <= 47) or (i >= 53 and i <= 57) or (i >= 63 and i <= 67) or (i >= 71 and i <= 73) or \
                        (i >= 81 and i <= 83) or (i >= 92 and i <= 93):
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

                elif imagemcount == 4:
                    if i == 23 or i == 33 or i == 34 or i == 54 or i == 55:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)

                elif imagemcount == 5:
                    if (i >= 0 and i <= 2) or (i >= 10 and i <= 12) or (i >= 20 and i <= 23) or (
                            i >= 29 and i <= 33) or (i >= 40 and i <= 43) or (i >= 49 and i <= 54) or (
                            i >= 58 and i <= 64) or (i >= 68 and i <= 75) or (i >= 78 and i <= 86) or (
                            i >= 88 and i <= 96) or (i >= 98 and i <= 99):
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                elif imagemcount == 6:
                    if i == 36 or i == 43 or (i >= 46 and i <= 47) or i == 53 or (i >= 56 and i <= 57) or i == 64 or\
                            (i >= 66 and i <= 67) or i == 74 or (i >= 76 and i <= 77) or (i >= 84 and i <= 87) or\
                            (i >= 89 and i <= 97) or i == 99:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                elif imagemcount == 7:
                    if (i >= 7 and i <= 9) or (i >= 16 and i <= 17) or i == 19 or (i >= 24 and i <= 29) or (
                            i >= 35 and i <= 39) or (i >= 45 and i <= 49) or (i >= 54 and i <= 59) or (
                            i >= 62 and i <= 63) or (i >= 65 and i <= 69) or (i >= 72 and i <= 79) or (
                            i >= 84 and i <= 87) or i == 96:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                elif imagemcount == 8:
                    if (i >= 49 and i <= 53) or (i >= 55 and i <= 58) or i == 60:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                elif imagemcount == 9:
                    if i == 23 or (i >= 30 and i <= 33) or (i >= 40 and i <= 43) or (i >= 50 and i <= 54) or (
                            i >= 60 and i <= 65) or (i >= 70 and i <= 77) or (i >= 80 and i <= 88) or (
                            i >= 90 and i <= 99):
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                elif imagemcount == 10:
                    if (i >= 32 and i <= 34) or i == 36 or i == 39 or (i >= 41 and i <= 44) or (i >= 46 and i <= 49) or (
                            i >= 51 and i <= 61) or (i >= 63 and i <= 65) or (i >= 70 and i <= 71) or (i >= 80 and i <= 81):
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 1
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    else:
                        img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                        # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                        img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
        # train_df.to_csv(csvfile_training, index=False)
        # teste = pd.read_csv(csvfile_training, dtype=str)

        # if train_df.equals(teste):
        #     print("igual")
    # SPLITING X and y DATASETS
    y = train_df['Target']
    # print(y.shape)

    X = train_df
    X.pop('Target')
    # print(X.shape)

    # TRAINING KNN
    knn_class = KNeighborsClassifier(n_neighbors=50)
    knn_class.fit(X, y)

    # fireImageDir = 'Dataset/Training and Validation/fire/*.jpg'
    # nofireImageDir = 'Dataset/Training and Validation/nofire/*.jpg'
    fireImageDir = 'Dataset/Testing/fire/*.jpg'
    nofireImageDir = 'Dataset/Testing/nofire/*.jpg'
    dir = [fireImageDir, nofireImageDir]

    ImageTest(knn_class, 'dataset/Testing/fire/abc162.jpg', True)

    # tn, fp, fn, tp = DirImageTest(knn_class, dir)
    #
    # fireImageDir = 'Dataset/create/fire/*.jpg'
    # test_df = manualTest(knn_class, fireImageDirTesting)
    #
    # X_test = test_df.copy()
    # X_test.pop('Target')
    #
    # test_df['Predict'] = knn_class.predict(X_test)
    # # test_df
    #
    # # %%
    # # COMPARE TARGET AND PREDICT
    # test_df.loc[test_df['Predict'] == test_df['Target'], 'Error'] = 0
    # test_df.loc[test_df['Predict'] != test_df['Target'], 'Error'] = 1
    # # test_df
    #
    # # %%
    # # CONFUSION MATRIX
    # tn, fp, fn, tp = confusion_matrix(test_df['Target'], test_df['Predict']).ravel()


    # tn = true positive ( with fire detect correctly)
    # fp = false positive ( with fire detect incorrectly)
    # fn = false negative ( with nofire detect incorrectly)
    # tp = true negative ( with nofire detect correctly)

    print("tn, fp, fn, tp ", tn, fp, fn, tp)

    # ACCURACY
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    print('\nAccuracy: ' + str(accuracy * 100))

    # TRUE POSITIVE RATE -RECALL
    TPR = tp / (tp + fn)
    print('\nTrue Positive Rate(Recall): ' + str(TPR * 100))

    # FALSE POSITIVE RATE -
    FPR = fp / (fp + tn)
    print('False Positive Rate: ' + str(FPR * 100))

    # PRECISION
    Precision = tp / (fp + tp)
    print('Precision Rate: ' + str(Precision * 100))
    # F1 SCORE
    F1 = 2 * (Precision * TPR) / (Precision + TPR)
    print('F1 Score: ' + str(F1 * 100))

    # POSITIVE LIKELIHOOD RATIO
    PLR = TPR / FPR
    print('Positive Likelihood Ratio: ' + str(PLR))

    stopTime = datetime.datetime.now()
    diftime = stopTime - starTime
    print("Encerando a simulação em: ", stopTime.strftime("%H:%M:%S"))
    print("Duração da simulação: ", diftime.total_seconds())