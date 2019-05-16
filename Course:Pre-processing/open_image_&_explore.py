# Importação das bibliotecas necessárias
import cv2
import numpy as np
from matplotlib import pyplot as plt

HEIGH = 360
WIDTH = 360

def edges_algorithms(img, detector = None):

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
img1 = cv2.imread('Img_test1.png') #RGB
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('Img_test2.png',0) #GRAY SCALE

#Resize
img1 = cv2.resize(img1, (WIDTH, HEIGH),interpolation=cv2.INTER_CUBIC)
img2 = cv2.resize(img2, (WIDTH, HEIGH), interpolation=cv2.INTER_CUBIC)

#Show histogram
#plt.hist(img1.ravel(),256,[0,256])
#plt.savefig('hist1.png')

#Remove noise
img1 = cv2.GaussianBlur(img1, (7,7),1)
img1 = cv2.equalizeHist(img1)

#Segmentation
ret, thresh = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

cv2.imshow('image', thresh)
cv2.waitKey(0)

#detect edges
img_edge = edges_algorithms(thresh, 'canny')

cv2.imshow('image', img_edge)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(img_edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img1, contours, 0, (0,255,0), 3)

for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 0), 1)
cv2.imshow('image', img1)
cv2.waitKey(0)