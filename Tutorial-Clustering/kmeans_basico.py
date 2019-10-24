#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tutorial K-means: definição de funções
@author: karlvandesman
"""

# *** Importar bibliotecas ***
from sklearn.datasets import make_blobs
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

#%%
# *** Inicializar centroides ***
def inicializacaoCentroides(X, K):
    '''Seleciona aleatoriamente clusters dentre os exemplos de X'''
	
    centroidesIniciais = X[np.random.choice(X.shape[0], K, replace=False), :]
    
    return centroidesIniciais

#%%
# *** Encontrar centroide mais próximo ***
def encontraCentroidesProximos(X, centroides):
    '''A partir da distância euclidiana, calcula-se o centroide mais próximo 
    para cada exemplo. Com isso, atribui-se a esse exemplo um grupo.
    # Retorna:
    # Vetor de comprimento n_exemplos com valores dos índices dos centroides
    '''
    
    indicesCentroidesProximos = np.zeros(len(X))
    
    for i in range(len(X)):     
        distancia = [ distance.euclidean(X[i], j) for j in centroides ]
        cluster = np.argmin(distancia)
        indicesCentroidesProximos[i] = cluster
            
    return indicesCentroidesProximos

#%%
# *** Calcular valor médio para os centroides ***
def calcularMediaCentroides(X, indicesGrupos, K):
    '''Dado os exemplos atribuídos a cada grupo, calcula-se a média
    para '''
    centroides = []
    
    for i in range(K):
        centroides.append(np.mean(X[indicesGrupos==i][:], axis=0))

    return np.vstack(centroides)

#%%
# *** Algoritmo K-means ***
def kMeans(X, K, num_iter, detalhado=False):
    '''A partir de um conjunto de dados e um número de grupos previamente
    definido, encontra-se o melhor agrupamento dos dados'''
    
    plt.figure()

    centroides = inicializacaoCentroides(X, K)
    plt.scatter(centroides[:, 0], centroides[:, 1], marker='X', 
                        alpha=0.4, c='m', s=150)
    
    # Loop de iteração para aproximação dos grupos aos centroides
    for _ in range(num_iter):
        # A cada iteração, atribui-se cada exemplo ao centroide mais próximo
        indicesGrupos = encontraCentroidesProximos(X, centroides)
        
        # Atualiza o valor dos centroides, e guarda-se os centroides antigos
        centroidesAntigos = centroides
        centroides = calcularMediaCentroides(X, indicesGrupos, K)
        
        # Plotar caminho de atualização de centroides se especificado
        if detalhado: plotDetalhado(centroides, centroidesAntigos)
        
        # O algoritmo para se não há diferença na atualização dos centroides
        if np.array_equal(centroidesAntigos, centroides): break
    
    plotKmeans(X, centroides, indicesGrupos)
    
    return centroides, indicesGrupos

#%%
# *** Plot dados agrupados por clusters ***
def plotKmeans(X, centroides, indicesGrupos):
    '''Plota um gráfico dos dados agrupados pelos centroides, com cor
    específica para cada grupo'''
    
    LABEL_COLOR_MAP = {0 : 'r', 1 : 'b', 2: 'y', 3:'g'}
    label_color = [LABEL_COLOR_MAP[l] for l in indicesGrupos]
    
    plt.scatter(X[:, 0], X[:, 1], s=6, c=label_color)
    plt.scatter(centroides[:, 0], centroides[:, 1], marker='*', c='k', s=200,
                alpha = 0.6)
    plt.show()

#%%
# *** Plot da diferença na atualização dos centroides ***
def plotDetalhado(centroides, centroidesAntigos):
    '''Plotar os centroides e o caminho feito após as atualizações'''
    
    K = centroides.shape[0]
    
    for j in range(K):
        plt.arrow(centroidesAntigos[j, 0], centroidesAntigos[j, 1], 
                  centroides[j, 0] - centroidesAntigos[j, 0], 
                  centroides[j, 1] - centroidesAntigos[j, 1])
    
    plt.scatter(centroides[:, 0], centroides[:, 1], marker='*', c='k', 
                s=200, alpha = 0.6)