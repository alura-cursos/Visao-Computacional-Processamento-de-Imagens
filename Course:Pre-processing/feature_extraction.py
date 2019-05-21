# Importação das bibliotecas necessárias
import cv2
import numpy as np
from matplotlib import pyplot as plt
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

    # Draw keypoints
    """ image = cv2.drawKeypoints(img_teste, pontos_chave, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('Feature Method - ORB', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() """

    return descritores

def salvarDescritores(descritores, caminho):
    descritores = descritores.reshape((1,descritores.size))

    arquivo=open(os.path.join(caminho, 'orb_descritores.csv'),'a')

    np.savetxt(arquivo, descritores, delimiter=',', fmt='%i')
    arquivo.close()

def main():
    caminhos = ['/home/suayder/Documents/alura/People-Detection-Image-classification/Course:Pre-processing/INRIAPerson_Dataset/Train/positivos/',
    '/home/suayder/Documents/alura/People-Detection-Image-classification/Course:Pre-processing/INRIAPerson_Dataset/Train/negativos/']
    
    for caminho in caminhos:
        # r=raiz, d=diretorios, a = arquivos
        for r, d, a in os.walk(caminho):
            for arquivo in a:
                if '.png' in arquivo:
                    descritores = getDescritores(os.path.join(r, arquivo))
                    salvarDescritores(descritores, caminho)

if __name__ == "__main__":
    main()