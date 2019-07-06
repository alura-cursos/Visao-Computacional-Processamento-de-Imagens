# Importação das bibliotecas necessárias

import os
import shutil

try:
    os.mkdir('dadosImagem')
except FileExistsError:
    pass
try:
    os.mkdir('dadosImagem/Treinamento')
except FileExistsError:
    pass
try:
    os.mkdir('dadosImagem/Teste')
except FileExistsError:
    pass

for name in list(['pos','neg']):
    try:
        os.mkdir('dadosImagem/Treinamento/'+name)
    except FileExistsError:
        pass

    try:
        os.mkdir('dadosImagem/Teste/'+name)
    except FileExistsError:
        pass

treinamento_caminho = ['INRIAPerson/Train','dadosImagem/Treinamento']
teste_caminho = ['INRIAPerson/Test','dadosImagem/Teste']
print("-----------------\nAguarde...\n-----------------")
for raiz, diretorios, arquivos in os.walk(treinamento_caminho[0]):
    for diretorio in diretorios:
        counter = 0
        n_caminho = os.path.join(raiz, diretorio)
        caminho_salvar = os.path.join(treinamento_caminho[1], diretorio)

        for n_raiz, n_diretorios, n_arquivos in os.walk(n_caminho):
            for arquivo in n_arquivos:
                if counter >= 400:
                    break
                if '.png' in arquivo:
                    try:
                        shutil.copyfile(os.path.join(n_raiz, arquivo), os.path.join(caminho_salvar, arquivo))
                        counter+=1
                    except Exception as e:
                        print(str(e))

for raiz, diretorios, arquivos in os.walk(teste_caminho[0]):
    for diretorio in diretorios:
        counter = 0
        n_caminho = os.path.join(raiz, diretorio)
        caminho_salvar = os.path.join(teste_caminho[1], diretorio)

        for n_raiz, n_diretorios, n_arquivos in os.walk(n_caminho):
            for arquivo in n_arquivos:
                if counter >= 100:
                    break
                if '.png' in arquivo:
                    try:
                        shutil.copyfile(os.path.join(n_raiz, arquivo), os.path.join(caminho_salvar, arquivo))
                        counter+=1
                    except Exception as e:
                        print(str(e))
print("\n\n\n-----------------\n\Pronto!\n-----------------")