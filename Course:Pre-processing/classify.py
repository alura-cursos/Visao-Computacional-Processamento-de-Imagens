# Importação das bibliotecas necessárias
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import os

def getDescritores(img_caminho):
    ALTURA = 360
    LARGURA = 360

    # Ler imagens
    img_teste = cv2.imread(img_caminho) #RGB
    img_cinza = cv2.cvtColor(img_teste, cv2.COLOR_BGR2GRAY)
    #img_teste = cv2.imread('Img_test1.png',0) #GRAY SCALE

    # Redimensionar
    img_redimencionada = cv2.resize(img_cinza, (LARGURA, ALTURA),interpolation=cv2.INTER_CUBIC)

    # Remover o ruído (suavizar a imagem)
    img_suavizada = cv2.GaussianBlur(img_redimencionada, (7,7),1)
    img_equalizada = cv2.equalizeHist(img_suavizada)

    orb = cv2.ORB_create(nfeatures = 400)

    # Determinar key points
    pontos_chave = orb.detect(img_equalizada, None)

    pontos_chave, descritores = orb.compute(img_equalizada, pontos_chave)

    return descritores

def carregarDescritor(caminho):

    print(type(np.loadtxt(os.path.join(caminho, 'orb_descritores.csv'), delimiter=',')))
    exit(0)

def main():

    caminhos = ['/home/suayder/Documents/alura/People-Detection-Image-classification/Course:Pre-processing/INRIAPerson_Dataset/Train/positivos/',
    '/home/suayder/Documents/alura/People-Detection-Image-classification/Course:Pre-processing/INRIAPerson_Dataset/Train/negativos/']

    for caminho in caminhos:
        carregarDescritor(caminho)
    #knn = KNeighborsClassifier(n_neighbors=5)
    #knn.fit()

if __name__ == "__main__":
    main()