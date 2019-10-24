#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tutorial K-means: exemplo de compressão de imagem
@author: karlvandesman
"""

# *** Importar bibliotecas ***
from skimage import io
from sklearn.cluster import KMeans
import numpy as np
#import kmeans_basico

#%%
# *** Carregar a imagem desejada ***
imagem = io.imread('imagens/cloud_service.png')
#io.imshow(imagem)
#io.show()

#%% 
# Obtendo dimensões da imagem
linhas = imagem.shape[0]
colunas = imagem.shape[1]

# "Desenrolar" a imagem, colocando um vetor para cada cor
imagem_est = imagem.reshape(linhas*colunas, 3)

#%% 
# *** Aplicação do K-means ***
modeloKmeans = KMeans(n_clusters=16, n_init=1, max_iter=10)

# "Treinar" o modelo
modeloKmeans.fit(imagem_est)

#%% 
# *** Converter dados em imagem ***
clusters = np.asarray(modeloKmeans.cluster_centers_, dtype=np.uint8) 
labels = np.asarray(modeloKmeans.labels_, dtype=np.uint8)
imagem_comprimida = clusters[labels.reshape(linhas, colunas)]

#%%
# *** Apresentar a imagem e salvar ***
io.imshow(imagem_comprimida)
io.show()

#np.save('clusters.npy', clusters)    
io.imsave('imagens/imagem_comprimida.png', imagem_comprimida);