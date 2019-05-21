# Importação das bibliotecas necessárias
import cv2
import numpy as np
from matplotlib import pyplot as plt

ALTURA = 360
LARGURA = 360

def detectarBorda(img, detector = None):

    if detector == 'laplace':
        laplacian = cv2.Laplacian(img, cv2.CV_64F,5)
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(laplacian)
        return laplacian

    elif detector == 'canny':
        canny = cv2.Canny(thresh, 100,200)
        canny = np.absolute(canny)
        canny = np.uint8(canny)
        return canny
    else:
        print("Escolha um algoritmo para detecção: laplace ou canny")
        return None

# Ler imagens
img_teste = cv2.imread('Img_test1.png') #RGB
img_cinza = cv2.cvtColor(img_teste, cv2.COLOR_BGR2GRAY)
#img_teste = cv2.imread('Img_test1.png',0) #GRAY SCALE

#Resize
img_redimencionada = cv2.resize(img_cinza, (LARGURA, ALTURA),interpolation=cv2.INTER_CUBIC)

#Show histogram
#plt.hist(img1.ravel(),256,[0,256])
#plt.savefig('hist1.png')

#Remover o ruído (suavizar a imagem)
img_suavizada = cv2.GaussianBlur(img_redimencionada, (7,7),1)
img_equalizada = cv2.equalizeHist(img_suavizada)

#Segmentação
retangulos, img_segmentada = cv2.threshold(img_equalizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

cv2.imshow('image', img_segmentada)
cv2.waitKey(0)

#Detectar bordas
img_bordas = detectarBorda(img_segmentada, 'canny')

cv2.imshow('image', img_bordas)
cv2.waitKey(0)

contornos, hierarquia = cv2.findContours(img_bordas,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_teste, contornos, 0, (0,255,0), 3)

for c in contornos:
    (x, y, l, a) = cv2.boundingRect(c)
    cv2.rectangle(img_teste, (x, y), (x + l, y + a), (0, 0, 0), 1)
cv2.imshow('image', img_teste)
cv2.waitKey(0)