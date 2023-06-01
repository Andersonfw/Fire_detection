#%%
import glob

import pandas as pd
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


'''
        DIRETÓRIOS
'''
fireImageDirTest = 'Dataset/Testing/fire/*.jpg'
fireImageDirTraining = 'Dataset/Training and Validation/fire/*.jpg'
nofireImageDirTest = 'Dataset/Testing/nofire/*.jpg'
nofireImageDirTraining = 'Dataset/Training and Validation/nofire/*.jpg'

# %%
# CREATING TRAIN DATASET FOR FIRE SAMPLES
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
# train_df

#%%
# SPLITING X and y DATASETS
y = train_df['Target']
print(y.shape)

X = train_df
X.pop('Target')
print(X.shape)

# %%
# TRAINING KNN
knn_class = KNeighborsClassifier(n_neighbors=50)
knn_class.fit(X,y)


# %%
# CREATING TEST DATASET FOR FIRE SAMPLES
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

# %%
