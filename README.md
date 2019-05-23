# People-Detection-Image-classification

Repositório com algoritmos de visão computacional - aulas alura

Para o projeto usaremos a base de dados [INRIA Person](http://pascal.inrialpes.fr/data/human/). Nela o objetivo é encontrar imagens que possuam pessoas.

O projeto é composto por dois arquivos principais ``` Feature_extraction.py ``` e ``` classify.py ```. No primeiro é utilizado o algoritmo *ORB* para extração de descritores da imagem e estes descritores processados com o algoritmo *Bag of Visual Words* que gerará um dicionário de palavras e assim irá gerar novas caracteristicas a serem quantizadas em um vetor de 255 posições que será a base de dados para teste e treinamento do modelo.

No arquivo ``` classify.py ``` é utilizado KNN para classificar cada imagem se possui ou não pessoa na imagem e então é verificado a acurácia geral do modelo.
