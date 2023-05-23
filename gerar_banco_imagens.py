# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:48:31 2021

@author: jeans
"""

import cv2

#Nome do arquivo a ser lido
filename = 'TCC_Nat_mod.jpg'

#Le o arquivo já em escala de cinza
original = cv2.imread(filename,0)

#Faz um recorte da imagem, aqui o bom é pensar no tamanho da imagem. O bom é pensar em números múltilplos de 20.
#De repente mantemos o tamanho original e diminuimos até o primeiro multiplo de 20.
original = original[0:700, 0:560]

cv2.imshow('Imagem Original', original)

test_image = original

#Cores para demarcar as janelas.
color = (255, 0, 0)
color1 = (0, 255, 0)
thickness = 2

# Define o tamanho da janela
windowsize_r = 20 #altura
windowsize_c = 20 #largura

isInteresse = []
imagens = []
subWindowsPos = []

for r in range(0,test_image.shape[0], windowsize_r):
    for c in range(0,test_image.shape[1], windowsize_c):
        window = test_image[r:r+windowsize_r,c:c+windowsize_c]
        
        start_point = (c,r)
        end_point = (c+windowsize_c,r+windowsize_r)
        
        original = cv2.rectangle(original, start_point, end_point, color1, thickness)
        
        window = test_image[r:r+windowsize_r, c:c+windowsize_c]
        
        #Região maior na direita. Vários retângulos.
        if (700 > c > 320) and (700 > r ):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
        
        #Perto da região maior.
        elif (340 > c > 300) and (580 > r ):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
        
        #Quadrados individuais. A partir daqui é trabalho de formiguinha. Tem que marcar todas as copas de árvore.
        elif (240 == c) and (r == 300):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (240 == c) and (r == 320):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (240 == c) and (r == 380):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (240 == c) and (r == 400):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (240 == c) and (r == 420):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (240 == c) and (r == 540):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
        
        elif (220 == c) and (r == 00):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (220 == c) and (r == 20):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (220 == c) and (r == 100):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
        
        elif (200 == c) and (r == 40):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (200 == c) and (r == 300):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (200 == c) and (r == 280):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
        
        elif (180 == c) and (r == 0):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
        
        elif (180 == c) and (r == 20):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
        
        elif (180 == c) and (r == 80):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
        
        elif (180 == c) and (r == 100):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (180 == c) and ( r == 140):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (180 == c) and ( r == 160):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
        
        elif (180 == c) and ( r == 180):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (180 == c) and ( r == 300):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (180 == c) and ( r == 340):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (160 == c) and ( r == 0):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (160 == c) and ( r == 20):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (160 == c) and ( r == 100):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (160 == c) and ( r == 160):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (160 == c) and ( r == 180):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (160 == c) and ( r == 340):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (140 == c) and ( r == 20):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (120 == c) and ( r == 20):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (120 == c) and ( r == 100):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (120 == c) and ( r == 180):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (120 == c) and ( r == 200):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (120 == c) and ( r == 260):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (120 == c) and ( r == 360):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (120 == c) and ( r == 580):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (100 == c) and ( r == 180):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (100 == c) and ( r == 200):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (100 == c) and ( r == 280):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (100 == c) and ( r == 360):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (80 == c) and ( r == 140):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (80 == c) and ( r == 160):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (80 == c) and ( r == 200):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (80 == c) and ( r == 360):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (60 == c) and ( r == 160):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (40 == c) and ( r == 360):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (40 == c) and ( r == 660):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (20 == c) and ( r == 320):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (20 == c) and ( r == 340):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (20 == c) and ( r == 360):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (20 == c) and ( r == 380):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (20 == c) and ( r == 600):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (20 == c) and ( r == 620):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (20 == c) and ( r == 660):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (0 == c) and ( r == 320):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (0 == c) and ( r == 340):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
            
        elif (0 == c) and ( r == 360):
            original = cv2.rectangle(original, start_point, end_point, color, thickness)
            isInteresse.append(1)
        else:
            isInteresse.append(0)
            
        subWindowsPos.append((c,r))   
        imagens.append(window)


cv2.imshow('Marcacoes', original)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Salva a imagem com as marcacoes.
#cv2.imwrite(filename[0:-4] + 'marcacao_regioes.jpg', original)