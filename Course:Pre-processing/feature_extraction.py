# Importação das bibliotecas necessárias
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

def bovw_gerarDicionario(lista_descritores):

    kmeans = KMeans(n_clusters = 256)
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

    orb = cv2.ORB_create(nfeatures = 512)

    # Determinar key points
    pontos_chave = orb.detect(img_equalizada, None)

    pontos_chave, descritores = orb.compute(img_equalizada, pontos_chave)

    return descritores

def bovw_computarDescritores(descritores):
    algoritmo_knn = NearestNeighbors(n_neighbors=1)
    algoritmo_knn.fit(dicionario)
    res = algoritmo_knn.kneighbors(descritores, return_distance=False).flatten()
    histograma_caracteristicas = np.histogram(res, bins=np.arange(dicionario.shape[0]))[0]
    return histograma_caracteristicas

def salvarDescritores(descritores, caminho):
    descritores = descritores.reshape((1,descritores.size))

    arquivo=open(os.path.join(caminho, 'orb_descritores.csv'),'a')

    np.savetxt(arquivo, descritores, delimiter=',', fmt='%i')
    arquivo.close()

def bovw_salvarDicionario():
    global dicionario
    if dicionario is not None:
        #print("Shape dicionario antes: ", dicionario.shape)
        #dicionario = dicionario.reshape((1,dicionario.size))
        print("Shape dicionario: ", dicionario.shape)
        np.savetxt('dicionario.csv', dicionario, delimiter=',', fmt='%f')
    else:
        print("Dicionario is none")

def bovw_carregarDicionario():
    global dicionario
    dicionario = np.loadtxt('dicionario.csv', delimiter=',')

# Tenho que salvar o dicionario
# gerar as features
# salvar e então classificar
def main():
    global dicionario
    caminhos = ['/home/suayder/Documents/alura/People-Detection-Image-classification/Course:Pre-processing/INRIAPerson_Dataset/Train/positivos/',
    '/home/suayder/Documents/alura/People-Detection-Image-classification/Course:Pre-processing/INRIAPerson_Dataset/Train/negativos/']
    
    # Rotina para criação do dicionário de palavras virtuais
    descritores = np.empty((0,32), dtype=np.uint8)
    for caminho in caminhos:
        i = 0
        # r=raiz, d=diretorios, a = arquivos
        for r, d, a in os.walk(caminho):
            for arquivo in a:
                if i >= 400: #Somente 400 exemplos positivos e 400 exemplos negativos
                    break
                if '.png' in arquivo:
                    descritores = np.append(descritores, getDescritores(os.path.join(r, arquivo)), axis=0)
                    i+=1
 
    ## Bag of Visual words - gerar dicionário de palavras
    bovw_gerarDicionario(descritores)
    bovw_salvarDicionario()
    
    # Salvar histograma de descritores de exemplos positivos e negativos de cada imagem
    """ bovw_carregarDicionario() """

    for caminho in caminhos:
        i = 0
        # r=raiz, d=diretorios, a = arquivos
        for r, d, a in os.walk(caminho):
            for arquivo in a:
                if i >= 400: #Somente 400 exemplos positivos e 400 exemplos negativos
                    break
                if '.png' in arquivo:
                    descritor = getDescritores(os.path.join(r, arquivo))
                    hitograma_descritor = bovw_computarDescritores(descritor)
                    salvarDescritores(hitograma_descritor, caminho)
                    i+=1  

if __name__ == "__main__":
    main()

#https://stackoverflow.com/questions/23676365/opencv-orb-descriptor-how-exactly-is-it-stored-in-a-set-of-bytes
#https://www.kaggle.com/wesamelshamy/tutorial-image-feature-extraction-and-matching
#https://gurus.pyimagesearch.com/the-bag-of-visual-words-model/