# Importação das bibliotecas necessárias
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

def bovw_gerarDicionario(lista_descritores):

    kmeans = KMeans(n_clusters = 600)
    kmeans = kmeans.fit(lista_descritores)
    global dicionario
    dicionario = kmeans.cluster_centers_

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

def bovw_computarDescritores(descritores):
    algoritmo_knn = KNeighborsClassifier(n_neighbors=1)
    res = algoritmo_knn.fit(descritores, dicionario)
    print(res.shape)

def salvarDescritores(descritores, caminho):
    descritores = descritores.reshape((1,descritores.size))

    arquivo=open(os.path.join(caminho, 'orb_descritores.csv'),'a')

    np.savetxt(arquivo, descritores, delimiter=',', fmt='%i')
    arquivo.close()

def salvarDicionario():
    global dicionario
    if dicionario is not None:
        #print("Shape dicionario antes: ", dicionario.shape)
        #dicionario = dicionario.reshape((1,dicionario.size))
        print("Shape dicionario: ", dicionario.shape)
        np.savetxt('dicionario.csv', dicionario, delimiter=',', fmt='%f')
    else:
        print("Dicionario is none")

def main():
    caminhos = ['/home/suayder/Documents/alura/People-Detection-Image-classification/Course:Pre-processing/INRIAPerson_Dataset/Train/positivos/',
    '/home/suayder/Documents/alura/People-Detection-Image-classification/Course:Pre-processing/INRIAPerson_Dataset/Train/negativos/']
    descritores = np.empty((0,32), dtype=np.uint8)
    for caminho in caminhos:
        i = 0
        # r=raiz, d=diretorios, a = arquivos
        for r, d, a in os.walk(caminho):
            for arquivo in a:
                if i >= 200:
                    break
                if '.png' in arquivo:
                    descritores = np.append(descritores, getDescritores(os.path.join(r, arquivo)), axis=0)
                    i+=1
    # Bag of Visual words
    bovw_gerarDicionario(descritores)
    salvarDicionario()
    #bagOfVisualWords(descritores)
    #salvarDescritores(descritores, caminho)

if __name__ == "__main__":
    main()

#https://stackoverflow.com/questions/23676365/opencv-orb-descriptor-how-exactly-is-it-stored-in-a-set-of-bytes
#https://www.kaggle.com/wesamelshamy/tutorial-image-feature-extraction-and-matching
#https://gurus.pyimagesearch.com/the-bag-of-visual-words-model/