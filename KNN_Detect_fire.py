"""
Created on maio 30 21:04:36 2023

@author: Ânderson Felipe Weschenfelder
"""


import csv
import locale
import datetime
import time
import os
import glob
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier  # Para classificação
from sklearn.ensemble import RandomForestRegressor  # Para regressão
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # Para classificação
from sklearn.metrics import mean_squared_error  # Para regressão
from sklearn.naive_bayes import GaussianNB  # Para o Naive Bayes Gaussiano
from sklearn.naive_bayes import MultinomialNB  # Para o Naive Bayes Multinomial

from sklearn.metrics import confusion_matrix
import submatrix as subM
import Evalution_Test as Ev
import boxplot as bpl


def saveresults(accuracy, TPR, FPR, Precision, F1, PLR, testeTesting, dfresult):

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


'''
        DIRETÓRIOS
'''
csvfile_training = "training_df.csv"
fireImageDirTraining = 'Dataset/create/Training/*.jpg'
fireImageDirTesting = 'Dataset/create/Testing/*.jpg'
nofireImageDir = 'Dataset/create/nofire/*.jpg'
saveImageDir = 'Dataset/create/save/'
csvImagesClassified = 'Excel_data/imagesclassified.csv'
csvtestResult = "csvtestResult.csv"
dfresult = pd.DataFrame()
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

'''
        VARIÁVEIS DE SIMULAÇÃO
'''
blue = 0  # index da matriz blue da imagem
green = 1  # index da matriz green da imagem
red = 2  # index da matriz red da imagem
listClassSubmatrix = []  # array de classes da submatrizes
listClassSubmatrixY = []  # array de classes da submatrizes

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
    timestart = time.time()
    print("iniciando simulação em: ", starTime.strftime("%H:%M:%S"))
    '''
            TREINAMENTO COM IMAGENS DE TESTE
    '''
    csvimages = pd.read_csv(csvImagesClassified, delimiter=';')
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
            imgy = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            # height, width, dim = img.shape
            # Divide a imagem em 100 submatrizes
            listClassSubmatrix = subM.dividerImage(img, submatriz_height, submatriz_width, submatriz_length)
            listClassSubmatrixY = subM.dividerImage(imgy, submatriz_height, submatriz_width, submatriz_length)
            # Cria um dataframe identificando cada submatriz com fogo (Target=1) ou sem fogo (Target=o)
            # cada linha representa uma submatrix e contém 25x25x3 colunas, sendo os PIXEL. Ainda é adicionado uma coluna de 'TARGET' para identifcar fogo ou não
            for n in range(0, csvimages.shape[0], 100):
                if image_name == csvimages.at[n, 'imagem']:
                    for k in range(100):
                        # imgy = subM.mount_Dataframe(listClassSubmatrixY[k])
                        img_df = subM.mount_Dataframe(listClassSubmatrix[k])
                        # img_df = pd.concat([imgy, img_df], ignore_index=True, axis=1)
                        # img_df = subM.mount_Dataframe(listClassSubmatrixY[k])
                        if csvimages.at[n + k, 'isfire'] == 1:
                            img_df['Target'] = 1
                        else:
                            img_df['Target'] = 0
                        train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
                    break

    mountdatasettrain = time.time()

    print("Mount dataset Training time: {:.2f} milissegundos".format((mountdatasettrain - timestart)*1000))
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
    timetrainingKNN = time.time()
    # knn_class = GaussianNB()  # Para o Naive Bayes Gaussiano
    # knn_class = MultinomialNB()  # Para o Naive Bayes Multinomial
    # knn_class = RandomForestClassifier()  # Para classificação
    # knn_class = RandomForestRegressor()  # Para regressão
    knn_class = KNeighborsClassifier(n_neighbors=50)
    # knn_class = KNeighborsClassifier(n_neighbors=50, weights='distance', p=1, algorithm='ball_tree')
    # knn_class = svm.SVC(kernel='rbf')
    # knn_class = svm.SVC(kernel='poly', degree=6, C = 0.3)
    knn_class.fit(X, y)
    print("Training KNN time: {:.2f} milissegundos".format((timetrainingKNN - mountdatasettrain)*1000))


    fireImageDir = 'Dataset/Testing/fire/*.jpg'
    nofireImageDir = 'Dataset/Testing/nofire/*.jpg'
    dir = [fireImageDir, nofireImageDir]
    #

    # Ev.ImageTest(knn_class, 'dataset/Testing/fire/abc162.jpg', submatriz_height, submatriz_width, False, save=False)
    timepredictimagealone = time.time()
    print("Predic 1 image time: {:.2f} milissegundos".format((timepredictimagealone - timetrainingKNN) * 1000))

    # Ev.ImageTest(knn_class, 'C:/Users/ander/Downloads/teste.png', submatriz_height, submatriz_width, True)
    # Ev.ImageTest(knn_class, 'dataset/Testing/fire/abc080.jpg', submatriz_height, submatriz_width, True)
    # Ev.ImageTest(knn_class, 'dataset/Training and Validation/fire/fire_0004.jpg', submatriz_height, submatriz_width, True)
    # Ev.ImageTest(knn_class, 'dataset/Testing/fire/abc132.jpg', submatriz_height, submatriz_width, plot=True, save=True)
    Ev.ImageTest(knn_class, 'dataset/Testing/fire/abc116.jpg', submatriz_height, submatriz_width, plot=True, save=False)

    # tn, fp, fn, tp = Ev.DirImageTest(knn_class, dir, submatriz_height, submatriz_width)

    # Ev.TestEvaluation(tn, fp, fn, tp)

    tn, fp, fn, tp, test_df = Ev.manualTest(knn_class, fireImageDirTesting, submatriz_height, submatriz_width, csvImagesClassified)
    timetesting = time.time()
    print("testing 26 images time: {:.2f} milissegundos".format((timetesting - timepredictimagealone) * 1000))
    accuracy, TPR, FPR, Precision, F1, PLR = Ev.TestEvaluation(tn, fp, fn, tp)

    # plt.figure()
    # bpl.plotboxplot(test_df)
    # plt.show()

    saveresults(accuracy, TPR, FPR, Precision, F1, PLR, testeTesting, dfresult)

    stopTime = datetime.datetime.now()
    diftime = stopTime - starTime
    print("Encerando a simulação em: ", stopTime.strftime("%H:%M:%S"))
    print("Duração da simulação: ", diftime.total_seconds())