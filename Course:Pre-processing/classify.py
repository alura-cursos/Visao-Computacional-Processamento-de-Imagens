# Importação das bibliotecas necessárias

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import os
from feature_extraction import getDescritores, bovw_computarDescritores, bovw_carregarDicionario

def carregarDescritores(caminho):

    descritores = np.loadtxt(os.path.join(caminho, 'orb_descritores.csv'), delimiter=',')
    return descritores
    

def main():

    caminhos = ['/home/suayder/Documents/alura/People-Detection-Image-classification/Course:Pre-processing/INRIAPerson_Dataset/Train/positivos/',
    '/home/suayder/Documents/alura/People-Detection-Image-classification/Course:Pre-processing/INRIAPerson_Dataset/Train/negativos/']

    #Carregar os descritores salvos
    descritores = np.empty((0,255))
    for caminho in caminhos:
        descritores = np.append(descritores, carregarDescritores(caminho), axis=0)

    #KNN para classificar as imagens

    rotulos = np.ones(400, dtype=np.uint8) #Rotular as primeiras 400 caracteristicas como 1
    rotulos = np.append(rotulos, np.zeros(400)) #Rotular as outras 400 como zero

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(descritores, rotulos)

    # Ler imagens. 50 positivas e 50 negativas

    caminhos_test = ['/home/suayder/Documents/alura/People-Detection-Image-classification/Course:Pre-processing/INRIAPerson_Dataset/Test/positivos/',
    '/home/suayder/Documents/alura/People-Detection-Image-classification/Course:Pre-processing/INRIAPerson_Dataset/Test/negativos/']

    img_testDescritores = np.empty((0,255), dtype=np.uint8)
    for caminho in caminhos_test:
        i = 0
        # r=raiz, d=diretorios, a = arquivos
        for r, d, a in os.walk(caminho):
            for arquivo in a:
                if i >= 100:
                    break
                if '.png' in arquivo:
                    img_descritor = getDescritores(os.path.join(r, arquivo))
                    img_descritor = bovw_computarDescritores(img_descritor)
                    img_dim_expandida = np.expand_dims(img_descritor, axis=0)
                    img_testDescritores = np.append(img_testDescritores, img_dim_expandida, axis=0)
                    i+=1
    rotulos_teste = np.concatenate((np.ones(100, dtype=np.uint8), np.zeros(100, dtype=np.uint8))) 
    print(knn.score(img_testDescritores, rotulos_teste))

if __name__ == "__main__":
    global dicionario
    bovw_carregarDicionario()
    main()