"""
Created on junho 02 12:59:03 2023

@author: Ânderson Felipe Weschenfelder
"""
import os
import glob
import cv2 as cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import submatrix as subM


def TestEvaluation(tn, fp, fn, tp):
    '''
    Calc the parameters to evaluation the test

    INPUT arguments
    tn : true negative ( with nofire detect correctly)
    fp : false positive ( with nofire detect incorrectly)
    fn : false negative ( with fire detect incorrectly)
    tp : true positive ( with fire detect correctly)
    OUTPUT
    none
    '''
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

    return accuracy, TPR, FPR, Precision, F1, PLR

def ImageTest(knn, imagename, submatriz_height, submatriz_width, plot=None, save = None):

    '''
    Detect fire in submatriz os an image

    INPUT arguments
    knn     :  knn class trained
    imagename    :  path to image
    submatriz_height   :  Height of submatriz
    submatriz_width :  Width of submatriz
    plot :  Plot of the result
    OUTPUT
    teste	: array of result from knn Predict
     '''
    submatriz_length = (submatriz_height, submatriz_width)
    img_test = cv2.imread(imagename, cv2.IMREAD_COLOR)
    # img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2YCrCb)
    height, width, dim = img_test.shape
    listClassSubmatrixTest, test_df = subM.dividerImage(img_test, submatriz_height, submatriz_width, submatriz_length,
                                                        True)
    teste = knn.predict(test_df)
    # print("Predict Result: ", teste)
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
        if inicioy == width:
            inicioy = 0
            iniciox += submatriz_height
        if iniciox == height:
            iniciox = 0
    # param_image = cv2.cvtColor(param_image, cv2.COLOR_YCrCb2BGR)
    if plot:
        cv2.imshow("Imagem Original", img_test)
        cv2.imshow("Imagem cortada", param_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save:
        image_name = "cort_" + os.path.basename(imagename)
        cv2.imwrite(image_name,param_image)

    return teste


def manualTest(knn, dir, submatriz_height, submatriz_width, csvimages):
    '''
    Detect fire in submatriz and generate parameters with KNN to evaluate the method

    INPUT arguments
    knn     :  knn class trained
    dir    :  directory to images test
    submatriz_height   :  Height of submatriz
    submatriz_width :  Width of submatriz
    OUTPUT
    tn : true negative ( with nofire detect correctly)
    fp : false positive ( with nofire detect incorrectly)
    fn : false negative ( with fire detect incorrectly)
    tp : true positive ( with fire detect correctly)
    '''
    submatriz_length = (submatriz_height, submatriz_width)
    imagemcount = 0
    files_list = glob.glob(dir)
    print("\r\n\n\ntest of detecting the position of fire in an image according to an images that is knows its true position")
    print("Test of images in", dir)
    test_df = pd.DataFrame()  # Dataframe para salvar os dados de cada submatriz
    csvimages = pd.read_csv(csvimages, delimiter=';')

    for files in files_list:
        imagemcount += 1
        image_name = os.path.basename(files)
        img_test = cv2.imread(files)
        imgy = cv2.cvtColor(img_test, cv2.COLOR_BGR2YCrCb)
        # Divide a imagem em 100 submatrizes
        listClassSubmatrix = subM.dividerImage(img_test, submatriz_height, submatriz_width, submatriz_length)
        listClassSubmatrixY = subM.dividerImage(imgy, submatriz_height, submatriz_width, submatriz_length)

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
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                break
        # Cria um dataframe identificando cada submatriz com fogo (Target=1) ou sem fogo (Target=o)
        # cada linha representa uma submatrix e contém 25x25x3 colunas, sendo os PIXEL. Ainda é adicionado uma coluna de 'TARGET' para identifcar fogo ou não

    X_test = test_df.copy()
    ret_df = test_df.copy()
    print(X_test.shape)
    X_test.pop('Target')

    test_df['Predict'] = knn.predict(X_test)

    test_df.loc[test_df['Predict'] == test_df['Target'], 'Error'] = 0
    test_df.loc[test_df['Predict'] != test_df['Target'], 'Error'] = 1

    tn, fp, fn, tp = confusion_matrix(test_df['Target'], test_df['Predict']).ravel()

    return tn, fp, fn, tp, ret_df

def DirImageTest(knn, dir,submatriz_height, submatriz_width):
    '''
    Detect fire in images and generate parameters to evaluate the method manually

    INPUT arguments
    knn     :  knn class trained
    dir    :  directory to images test
    submatriz_height   :  Height of submatriz
    submatriz_width :  Width of submatriz
    OUTPUT
    tn : true negative ( with nofire detect correctly)
    fp : false positive ( with nofire detect incorrectly)
    fn : false negative ( with fire detect incorrectly)
    tp : true positive ( with fire detect correctly)
    '''
    submatriz_length = (submatriz_height, submatriz_width)
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    imagemcount = 0
    detectCorrectcount = 0
    detectIncorrectcount = 0
    print("\r\n\n\ntest of detect images with fire")
    for i in range(2):
        files_list = glob.glob(dir[i])
        print("Test of images in", dir[i])
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
                    # print(image_name)
                else:
                    detectCorrectcount += 1
                    tn += 1

    print("De ", imagemcount, " foram detectadas ", detectCorrectcount, " imagens corretamente")
    print("% de acerto: ", (detectCorrectcount / imagemcount) * 100)
    return tn, fp, fn, tp