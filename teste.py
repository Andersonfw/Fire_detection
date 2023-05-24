"""
Created on maio 16 22:53:26 2023

@author: Ânderson Felipe Weschenfelder
"""
import numpy as np

# Tamanho das matrizes
tamanho = (4, 4)

# Criar array de matrizes vazio
array_matrizes = np.zeros((10,) + tamanho)

# Laço for para criar e incrementar as matrizes
for i in range(10):
    # Criar matriz
    # matriz = (i,i*10)

    # Adicionar matriz ao array
    array_matrizes[i, 1, 2] = i

print(array_matrizes[9])
