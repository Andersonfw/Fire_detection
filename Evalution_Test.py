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


def ImageTest(knn, imagename, submatriz_height, submatriz_width, plot=None):

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


def manualTest(knn, dir, submatriz_height, submatriz_width):
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
    for files in files_list:
        imagemcount += 1
        image_name = os.path.basename(files)
        img_test = cv2.imread(files)
        # Divide a imagem em 100 submatrizes
        listClassSubmatrix = subM.dividerImage(img_test, submatriz_height, submatriz_width, submatriz_length)
        # Cria um dataframe identificando cada submatriz com fogo (Target=1) ou sem fogo (Target=o)
        # cada linha representa uma submatrix e contém 25x25x3 colunas, sendo os PIXEL. Ainda é adicionado uma coluna de 'TARGET' para identifcar fogo ou não
        for i in range(len(listClassSubmatrix)):
            img = listClassSubmatrix[i].matrix.copy()  # recebe a matriz do index i
            if image_name == "fire_0100.jpg":
                if i == 36 or i == 43 or (i >= 46 and i <= 48) or (i >= 56 and i <= 57) or \
                        (i >= 66 and i <= 67) or i == 74 or (i >= 76 and i <= 78) or (i >= 82 and i <= 87) or \
                        (i >= 89 and i <= 97) or i == 99:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)

            elif image_name == "fire_0101.jpg":
                if (i >= 7 and i <= 9) or (i >= 16 and i <= 19) or (i >= 26 and i <= 29) or (
                        i >= 35 and i <= 39) or (i >= 44 and i <= 49) or (i >= 54 and i <= 59) or (
                        i >= 62 and i <= 69) or (i >= 72 and i <= 79) or (
                        i >= 83 and i <= 87) or i == 89 or (i >= 92 and i <= 97):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            elif image_name == "fire_0014.jpg":
                if (1 <= i <= 2) or i == 8 or (i >= 11 and i <= 12) or i == 14 or (i >= 16 and i <= 19) or (
                        i >= 21 and i <= 22) \
                        or i == 24 or (i >= 26 and i <= 29) or (i >= 31 and i <= 39) or (i >= 41 and i <= 49) or i == 51 \
                        or (i >= 57 and i <= 59) or i == 62 or (i >= 65 and i <= 69) or (i >= 71 and i <= 79) \
                        or (i >= 81 and i <= 86) or i == 89 or (i >= 93 and i <= 97):

                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            if image_name == "fire_0008.jpg":
                if (i >= 35 and i <= 36) or (i >= 43 and i <= 46) or (i >= 54 and i <= 59) or (i >= 64 and i <= 69) or \
                        (i >= 75 and i <= 79) or (i >= 84 and i <= 87) or (i >= 92 and i <= 97):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)

            # elif image_name == "fire_0014.jpg":
            #     if (1 <= i <= 2) or i == 8 or (i >= 11 and i <= 12) or i == 14 or (i >= 16 and i <= 19) or (i >= 21 and i <= 22) \
            #             or i == 24 or (i >= 26 and i <= 29) or (i >= 31 and i <= 39) or (i >= 41 and i <= 49) or i == 51 \
            #             or (i >= 57 and i <= 59) or i == 62 or (i >= 65 and i <= 69) or (i >= 71 and i <= 79) \
            #             or (i >= 81 and i <= 86) or i == 89 or (i >= 93 and i <= 97):
            #
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 1
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #     else:
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 0
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)

            elif image_name == "fire_0048.jpg":
                if (i >= 6 and i <= 8) or (i >= 17 and i <= 18) or (i >= 26 and i <= 28) or i == 31 or i == 33 or (
                        i >= 35 and i <= 38) or \
                        i == 41 or (i >= 43 and i <= 48) or i == 51 or (i >= 53 and i <= 58) or (
                        i >= 61 and i <= 68) or (i >= 71 and i <= 74) or \
                        (i >= 81 and i <= 83) or (i >= 91 and i <= 93):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)


            elif image_name == "fire_0049.jpg":
                if (i >= 3 and i <= 4) or (i >= 12 and i <= 14) or (i >= 22 and i <= 24) or (i >= 32 and i <= 35) or \
                        (i >= 54 and i <= 56) or (i >= 65 and i <= 66) or i == 69 or (i >= 77 and i <= 79):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)


            elif image_name == "fire_0051.jpg":
                if (i >= 0 and i <= 3) or (i >= 10 and i <= 12) or i == 14 or (i >= 20 and i <= 24) or (
                        i >= 29 and i <= 34) or (i >= 38 and i <= 44) or (i >= 48 and i <= 54) or (
                        i >= 58 and i <= 65) or (i >= 69 and i <= 75) or (i >= 78 and i <= 99):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)

            # elif image_name == "fire_0100.jpg":
            #     if i == 36 or i == 43 or (i >= 46 and i <= 48) or (i >= 56 and i <= 57) or\
            #             (i >= 66 and i <= 67) or i == 74 or (i >= 76 and i <= 78) or (i >= 82 and i <= 87) or\
            #             (i >= 89 and i <= 97) or i == 99:
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 1
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #     else:
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 0
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #
            # elif image_name == "fire_0101.jpg":
            #     if (i >= 7 and i <= 9) or (i >= 16 and i <= 19) or (i >= 26 and i <= 29) or (
            #             i >= 35 and i <= 39) or (i >= 44 and i <= 49) or (i >= 54 and i <= 59) or (
            #             i >= 62 and i <= 69) or (i >= 72 and i <= 79) or (
            #             i >= 83 and i <= 87) or i == 89 or (i >= 92 and i <= 97):
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 1
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #     else:
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 0
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)

            elif image_name == "fire_0124.jpg":
                if (i >= 39 and i <= 61):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)

            elif image_name == "fire_0134.jpg":
                if i == 23 or (i >= 30 and i <= 33) or (i >= 40 and i <= 42) or (i >= 50 and i <= 54) or (
                        i >= 60 and i <= 66) or (i >= 70 and i <= 77) or (i >= 80 and i <= 99):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)

            elif image_name == "fire_0158.jpg":
                if i == 24 or (i >= 32 and i <= 34) or i == 36 or (i >= 41 and i <= 44) or (i >= 46 and i <= 49) or (
                        i >= 51 and i <= 60) or (i >= 63 and i <= 65) or (i >= 70 and i <= 71) or i == 80:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                #############################################################
            elif image_name == "fire_0005.jpg":
                if i == 32 or (i >= 51 and i <= 52) or i == 56 or i == 61 or (i >= 64 and i <= 66) or (
                        i >= 74 and i <= 76) or \
                        (i >= 82 and i <= 85) or i == 87 or (i >= 92 and i <= 94):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            elif image_name == "fire_0085.jpg":
                if i == 34 or (i >= 40 and i <= 45) or (i >= 50 and i <= 55) or (i >= 57 and i <= 66) or \
                        (i >= 73 and i <= 76) or (i >= 80 and i <= 81) or (i >= 90 and i <= 91):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            elif image_name == "fire_0138.jpg":
                if i == 30 or i == 40 or (i >= 50 and i <= 51) or (i >= 53 and i <= 54) or i == 57 or i == 60 or (
                        i >= 63 and i <= 64) or \
                        (i >= 66 and i <= 67) or i == 69 or (i >= 73 and i <= 77):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            elif image_name == "fire_0209.jpg":
                if i == 13 or i == 33 or (i >= 43 and i <= 44) or (i >= 53 and i <= 55) or (i >= 62 and i <= 67) or \
                        (i >= 72 and i <= 76) or (i >= 82 and i <= 87):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            elif image_name == "fire_0262.jpg":
                if i == 44 or i == 49 or i == 59 or i == 69:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            elif image_name == "fire_0308.jpg":
                if i == 45 or (i >= 54 and i <= 55) or i == 59 or i == 61 or (i >= 64 and i <= 66) or (
                        i >= 69 and i <= 70) \
                        or (i >= 72 and i <= 79) or (i >= 82 and i <= 89) or i == 99:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            elif image_name == "fire_0359.jpg":
                if (i >= 57 and i <= 59) or (i >= 67 and i <= 69) or (i >= 76 and i <= 79) or i == 90:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            elif image_name == "fire_0482.jpg":
                if i == 87 or i == 99 or (i >= 80 and i <= 83) or (i >= 89 and i <= 93) or (i >= 96 and i <= 97):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)

            elif image_name == "fire_0108.jpg":
                if i == 18  or i == 28 or (i >= 37 and i <= 38) or (i >= 47 and i <= 48):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            elif image_name == "fire_0169.jpg":
                if (i >= 30 and i <= 33) or (i >= 35 and i <= 36) or (i >= 46 and i <= 47) or i == 57:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)


                    ###########  HIGH FLAME FIRE##################
            elif image_name == "fire_0535.jpg":
                if (i >= 2 and i <= 3) or (i >= 6 and i <= 9) or (i >= 12 and i <= 13) or (i >= 16 and i <= 19) or (i >= 22 and i <= 23) \
                        or (i >= 26 and i <= 29) or (i >= 32 and i <= 33) or (i >= 36 and i <= 39) or (i >= 42 and i <= 43) \
                        or (i >= 46 and i <= 49) or (i >= 51 and i <= 52) or (i >= 56 and i <= 59) or (i >= 61 and i <= 62) \
                        or (i >= 65 and i <= 69) or (i >= 71 and i <= 73) or (i >= 75 and i <= 77) or (i >= 81 and i <= 87) or (i >= 91 and i <= 93):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)

            elif image_name == "fire_0593.jpg":
                if (i >= 2 and i <= 4) or (i >= 11 and i <= 14) or i == 17 or (i >= 20 and i <= 24) or (i >= 26 and i <= 27)\
                        or (i >= 30 and i <= 37) or (i >= 40 and i <= 47) or (i >= 50 and i <= 58) or (i >= 60 and i <= 79) \
                        or (i >= 85 and i <= 89) or (i >= 96 and i <= 99):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            elif image_name == "fire_0727.jpg":
                if (i >= 6 and i <= 9) or i == 16 or i == 19 or (i >= 22 and i <= 23) or (i >= 26 and i <= 29) or (i >= 32 and i <= 33) \
                        or (i >= 35 and i <= 39) or (i >= 42 and i <= 49) or (i >= 52 and i <= 59) or (i >= 65 and i <= 68) or (i >= 74 and i <= 79):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)

            elif image_name == "fire_0035.jpg":
                    # if (i >= 40 and i <= 43) or (i >= 47 and i <= 49) or (i >= 56 and i <= 59) \
                    #         or (i >= 67 and i <= 69) or (i >= 78 and i <= 79) or i == 89:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                    # else:
                    #     img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    #     # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    #     img_df['Target'] = 0
                    #     train_df = pd.concat([train_df, img_df], ignore_index=True, axis=0)
            elif image_name == "fire_0045.jpg":
                if (i >= 6 and i <= 61) or i == 63 or (i >= 65 and i <= 71) or i == 73 or (i >= 75 and i <= 79) \
                        or i == 82 or i == 86 or i == 93 or (i >= 92 and i <= 93) or (i >= 96 and i <= 99):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            elif image_name == "fire_0444.jpg":
                if (i >= 3 and i <= 9) or (i >= 13 and i <= 19) or (i >= 23 and i <= 29) or (i >= 33 and i <= 39) \
                        or (i >= 43 and i <= 49) or (i >= 54 and i <= 61) or (i >= 64 and i <= 71) or (
                        i >= 74 and i <= 81) \
                        or (i >= 85 and i <= 89) or (i >= 96 and i <= 99):
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 1
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
                else:
                    img_df = subM.mount_Dataframe(listClassSubmatrix[i])
                    # img_df = pd.DataFrame(img.reshape(-1)).transpose()
                    img_df['Target'] = 0
                    test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)

                    ###################################################
            # elif image_name == "fire_0718.jpg":
            #     if (i >= 14 and i <= 15) or (i >= 23 and i <= 24) or (i >= 43 and i <= 44) or \
            #             (i >= 52 and i <= 54) or (i >= 62 and i <= 65) or (i >= 72 and i <= 75) \
            #             or (i >= 82 and i <= 86) or (i >= 93 and i <= 97):
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 1
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #     else:
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 0
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)



            # elif image_name == "fire_0690.jpg":
            #     if (i >= 44 and i <= 46) or (i >= 50 and i <= 52) or (i >= 54 and i <= 56) or (
            #             i >= 60 and i <= 62):
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 1
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #     else:
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 0
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)

            # elif image_name == "fire_0697.jpg":
            #     if i == 37 or i == 46 or i == 56 or (i >= 60 and i <= 62) or (i >= 66 and i <= 67) or (
            #             i >= 71 and i <= 73):
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 1
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #     else:
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 0
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #
            # elif image_name == "fire_0817.jpg":
            #     if i == 10 or i == 37 or (i >= 42 and i <= 49) or (i >= 52 and i <= 59) or \
            #             (i >= 61 and i <= 67) or (i >= 71 and i <= 72):
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 1
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #     else:
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 0
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            # if imagemcount == 1:
            #     if i == 47 or i == 55 or i == 56 or i == 57 or i == 65 or i == 69 or i == 74 or i == 75 or i == 83 or i == 84 or i == 85 or i == 93 or i == 94:
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 1
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #     else:
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 0
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #
            # elif imagemcount == 2:
            #     if (i >= 10 and i <= 13) or (i >= 20 and i <= 23) or (i >= 30 and i <= 33) or (i >= 40 and i <= 44) or (
            #             i >= 52 and i <= 54) or (i >= 63 and i <= 69) or i == 57 or i == 59 or (
            #             i >= 73 and i <= 79) or (i >= 83 and i <= 89) or (i >= 98 and i <= 99):
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 1
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #     else:
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 0
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #
            # elif imagemcount == 3:
            #     if (i >= 71 and i <= 75):
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 1
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #     else:
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 0
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #
            # elif imagemcount == 4:
            #     if i == 36 or i == 37 or i == 45 or i == 63 or i == 72 or i == 82 or i == 92:
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 1
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #     else:
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 0
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #
            # elif imagemcount == 5:
            #     if i == 9 or (i >= 17 and i <= 19) or (i >= 24 and i <= 29) or (i >= 32 and i <= 39) or (
            #             i >= 41 and i <= 49) or (i >= 51 and i <= 59) or (i >= 61 and i <= 65) or (
            #             i >= 72 and i <= 75) or (i >= 83 and i <= 84):
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 1
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)
            #     else:
            #         img_df = subM.mount_Dataframe(listClassSubmatrix[i])
            #         # img_df = pd.DataFrame(img.reshape(-1)).transpose()
            #         img_df['Target'] = 0
            #         test_df = pd.concat([test_df, img_df], ignore_index=True, axis=0)


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