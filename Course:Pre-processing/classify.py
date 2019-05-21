# Importação das bibliotecas necessárias
import cv2
import numpy as np
from matplotlib import pyplot as plt

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

def main():
    #Implementar nearest neighbors for try get images

if __name__ == "__main__":
    main()