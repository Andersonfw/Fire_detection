"""
Created on maio 23 19:50:03 2023

@author: Ânderson Felipe Weschenfelder
"""
import os
import pandas as pd
import numpy as np
import cv2
from sklearn.metrics._scorer import metric
from sklearn.neighbors import KNeighborsClassifier

# %%
# CREATING TRAIN DATASET FOR FIRE SAMPLES
for i in range(904):
    caminho_arquivo = 'Dataset/Training and Validation/fire/fire_0'+str(i).zfill(3)+'.jpg'
    if os.path.exists(caminho_arquivo):
        img = cv2.imread(caminho_arquivo, cv2.IMREAD_COLOR)
        if img is not None:
            if i == 1:
                train_df_fire = pd.DataFrame(img.reshape(-1)).transpose()
            else:
                img_df = pd.DataFrame(img.reshape(-1)).transpose()
                train_df_fire = pd.concat([train_df_fire,img_df], ignore_index=True, axis=0)

train_df_fire['Target'] = 1 # TARGET VARIABLE TO FLAG WHEN THERE IS FIRE
#Target = 1 fire
#Target = 0 no fine

# %%
print(train_df_fire.shape)
train_df_fire.sample(n=15)

# %%
# CREATING TRAIN DATASET FOR NON-FIRE SAMPLES
for i in range(761):
    caminho_arquivo = 'Dataset/Training and Validation/nofire/nofire_0'+str(i).zfill(3)+'.jpg'
    if os.path.exists(caminho_arquivo):
        img = cv2.imread(caminho_arquivo, cv2.IMREAD_COLOR)
        if img is not None:
            if i == 1:
                train_df_nofire = pd.DataFrame(img.reshape(-1)).transpose()
            else:
                img_df = pd.DataFrame(img.reshape(-1)).transpose()
                train_df_nofire = pd.concat([train_df_nofire,img_df], ignore_index=True, axis=0)

train_df_nofire['Target'] = 0 # TARGET VARIABLE TO FLAG WHEN THERE IS NO FIRE
# %%
print(train_df_nofire.shape)
train_df_nofire.sample(n=15)
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
print("training")
knn_class = KNeighborsClassifier(n_neighbors=50)
knn_class.fit(X,y)

# %%
print("predict")
# PREDICT A NEW IMAGE, here you can just select a random one
#img_test = cv2.imread('dataset/Testing/fire/abc004.jpg', cv2.IMREAD_COLOR)
img_test = cv2.imread('Dataset/Testing/nofire/abc195.jpg', cv2.IMREAD_COLOR)
X_test = pd.DataFrame(img_test.reshape(-1)).transpose()

res = knn_class.predict(X_test)

print(metric.classification_report(X_test, res))

print("resultado",res)

cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
