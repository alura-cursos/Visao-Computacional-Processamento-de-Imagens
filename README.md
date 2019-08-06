# People-Detection-Image-classification

Repositório com o projeto do curso de introdução a visão computacional - Alura

O objetivo do projeto é classificar imagens que possual pessoas ou não. Para isto usaremos a base de dados [INRIA Person](http://pascal.inrialpes.fr/data/human/).

O projeto principal é composto por um único arquivo ``` Detecção_de_imagens_com_pessoas.ipynb ```. Neste possui toda a rotina de extração de características e classificação e todo o algoritmo desenvolvido no decorrer das aulas,incluindo exemplos e testes feitos.

Como estrutura geral do algoritmo de classificação é utilizado o algoritmo *ORB* para extração de descritores da imagem e estes descritores processados com o algoritmo *Bag of Visual Words* que gerará um dicionário de palavras e assim irá gerar novas caracteristicas a serem quantizadas em um vetor de 512 posições que será a base de dados para teste e treinamento do modelo.

Para classificação é utilizado KNN para classificar cada imagem se possui ou não pessoa na imagem e então é verificado a acurácia geral do modelo mostrando também a matriz de confusão para entendimento do problema e plotado o gráfico com os principais componetes ([PCA](https://medium.com/@aptrishu/understanding-principle-component-analysis-e32be0253ef0)) da base de dados para análise principalmente da separabilidade dos dados.

Como o projeto foi desenvolvido no [google colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) há algumas coisas que são específicas então se não forem executados no *google colab* pode ser que haja algum erro.

Há também outros branchs além do master em cada estão os códigos das respectivas aulas (respectivas ao nome dos branchs). O último commit de cada vídeo da aula é o commit válido para o que foi desenvolvido na aula.