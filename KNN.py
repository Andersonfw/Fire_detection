#%%
import locale
import datetime
import time
import os
import glob
import pandas as pd
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier  # Para classificação
from sklearn.ensemble import RandomForestRegressor  # Para regressão
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # Para classificação
from sklearn.metrics import mean_squared_error  # Para regressão
from sklearn.naive_bayes import GaussianNB  # Para o Naive Bayes Gaussiano
from sklearn.naive_bayes import MultinomialNB  # Para o Naive Bayes Multinomial

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
fireImageDirTest = 'Dataset/Testing/fire/*.jpg'
fireImageDirTraining = 'Dataset/Training and Validation/fire/*.jpg'
nofireImageDirTest = 'Dataset/Testing/nofire/*.jpg'
nofireImageDirTraining = 'Dataset/Training and Validation/nofire/*.jpg'
csvtestResult = "csvtestResultKNNJeh.csv"
csvfile_training = "KNNjehtrain.csv"
csvfile_testing = "KNNjehtest.csv"
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
dfresult = pd.DataFrame()

# %%
# CREATING TRAIN DATASET FOR FIRE SAMPLES
starTime = datetime.datetime.now()
if not os.path.exists(csvtestResult):
    columns = ['time', 'img_train_num', 'img_test_num', 'Features_num', 'tn', 'fp', 'fn', 'tp', 'accuracy', 'TPR',
               'FPR', 'Precision', 'F1', 'PLR']
    testeTesting = pd.DataFrame(columns=columns)
else:
    testeTesting = pd.read_csv(csvtestResult, delimiter=';')

dfresult['time'] = [starTime.strftime("%H:%M:%S")]
if os.path.exists(csvfile_training):

    teste = pd.DataFrame()
    teste = pd.read_csv(csvtestResult,delimiter=';')
    train_df = pd.DataFrame(teste.values)
    train_df = train_df.rename(columns={train_df.columns[-1]: 'Target'})
else:

    files_list = glob.glob(fireImageDirTraining)
    train_df_fire = pd.DataFrame()
    for files in files_list:
        img = cv2.imread(files, cv2.IMREAD_COLOR)
        img_df = pd.DataFrame(img.reshape(-1)).transpose()
        train_df_fire = pd.concat([train_df_fire,img_df], ignore_index=True, axis=0)

    train_df_fire['Target'] = 1 # TARGET VARIABLE TO FLAG WHEN THERE IS FIRE
    #Target = 1 fire
    #Target = 0 no fine

    # %%
    print(train_df_fire.shape)
    # train_df_fire.sample(n=15)

    # %%
    # CREATING TRAIN DATASET FOR NON-FIRE SAMPLES
    files_list = glob.glob(nofireImageDirTraining)
    train_df_nofire = pd.DataFrame()
    for files in files_list:
        img = cv2.imread(files, cv2.IMREAD_COLOR)
        img_df = pd.DataFrame(img.reshape(-1)).transpose()
        train_df_nofire = pd.concat([train_df_nofire,img_df], ignore_index=True, axis=0)

    train_df_nofire['Target'] = 0 # TARGET VARIABLE TO FLAG WHEN THERE IS NO FIRE
    # %%
    print(train_df_nofire.shape)
    # train_df_nofire.sample(n=15)
    # %%
    # CONCATENATING BOTH FIRE AND NON-FIRE DATASETS FOR TRAINING
    train_df = pd.concat([train_df_fire,train_df_nofire], ignore_index=True, axis=0)
    print(train_df.shape)
    # train_df.to_csv(csvfile_training, index=False, sep=';')
    # train_df

#%%
print(train_df.shape)
# SPLITING X and y DATASETS
y = train_df['Target']
print(y.shape)

X = train_df
X.pop('Target')
print(X.shape)

# %%
# TRAINING KNN
knn_class = GaussianNB()  # Para o Naive Bayes Gaussiano
# knn_class = MultinomialNB()  # Para o Naive Bayes Multinomial
# knn_class = RandomForestClassifier()  # Para classificação
# knn_class = RandomForestRegressor()  # Para regressão
# knn_class = KNeighborsClassifier(n_neighbors=80)
# knn_class = KNeighborsClassifier(n_neighbors=50, weights='distance', p=1, algorithm='ball_tree')
# knn_class = svm.SVC(kernel='rbf')
# knn_class = svm.SVC(kernel='poly', degree=6, C = 0.3)
knn_class.fit(X,y)


# %%
# CREATING TEST DATASET FOR FIRE SAMPLES
if os.path.exists(csvfile_testing):

    teste = pd.DataFrame()
    teste = pd.read_csv(csvtestResult,delimiter=';')
    test_df = pd.DataFrame(teste.values)
    test_df = test_df.rename(columns={test_df.columns[-1]: 'Target'})
else:
    files_list = glob.glob(fireImageDirTest)
    test_df_fire = pd.DataFrame()
    for files in files_list:
        img = cv2.imread(files, cv2.IMREAD_COLOR)
        img_df = pd.DataFrame(img.reshape(-1)).transpose()
        test_df_fire = pd.concat([test_df_fire,img_df], ignore_index=True, axis=0)

    test_df_fire['Target'] = 1 # TARGET VARIABLE TO FLAG WHEN THERE IS FIRE


    # %%
    print(test_df_fire.shape)
    # test_df_fire.sample(n=15)

    # %%
    # CREATING TEST DATASET FOR NON-FIRE SAMPLES
    files_list = glob.glob(nofireImageDirTest)
    test_df_nofire = pd.DataFrame()
    for files in files_list:
        img = cv2.imread(files, cv2.IMREAD_COLOR)
        img_df = pd.DataFrame(img.reshape(-1)).transpose()
        test_df_nofire = pd.concat([test_df_nofire,img_df], ignore_index=True, axis=0)

    test_df_nofire['Target'] = 0 # TARGET VARIABLE TO FLAG WHEN THERE IS NO FIRE

    # %%
    print(test_df_nofire.shape)
    # test_df_nofire.sample(n=15)

    # %%
    # CONCATENATING BOTH FIRE AND NON-FIRE DATASETS FOR TESTING
    test_df = pd.concat([test_df_fire,test_df_nofire], ignore_index=True, axis=0)
    # test_df.to_csv(csvfile_testing, index=False, sep=';')
    # test_df = test_df_nofire


print(test_df.shape)
# test_df

# %%
X_test = test_df.copy()
X_test.pop('Target')

test_df['Predict'] = knn_class.predict(X_test)
# test_df

# %%
# COMPARE TARGET AND PREDICT
test_df.loc[test_df['Predict'] == test_df['Target'], 'Error'] = 0
test_df.loc[test_df['Predict'] != test_df['Target'], 'Error'] = 1
# test_df

# %%
# CONFUSION MATRIX
tn, fp, fn, tp = confusion_matrix(test_df['Target'], test_df['Predict']).ravel()
print("tn, fp, fn, tp ", tn, fp, fn, tp)

# %%
# ACCURACY
accuracy = (tp+tn) / len(test_df)
print('\nAccuracy: '+str(accuracy*100))

# TRUE POSITIVE RATE -RECALL
TPR = tp / (tp + fn)
print('\nTrue Positive Rate(Recall): '+str(TPR*100))

# FALSE POSITIVE RATE -
FPR = fp / (fp + tn)
print('False Positive Rate: '+str(FPR*100))

# PRECISION
Precision = tp/(fp + tp)
print('Precision Rate: '+str(Precision*100))
# F1 SCORE
F1 = 2*(Precision*TPR)/(Precision+TPR)
print('F1 Score: '+str(F1*100))

# POSITIVE LIKELIHOOD RATIO
PLR = TPR / FPR
print('Positive Likelihood Ratio: '+str(PLR))

saveresults(accuracy, TPR, FPR, Precision, F1, PLR, testeTesting, dfresult)

# %%
