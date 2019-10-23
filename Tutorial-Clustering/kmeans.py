#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 2019

@author: karlvandesman
"""
# *** Importação de bibliotecas ***
from sklearn.datasets import make_blobs
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

#%%
# Algoritmo K-means

# *** Inicialização de Centroids ***
def kMeansInicCentroides(X, K):
	# Seleciona os centros dos clusters iniciais
	
    centroidesIniciais = X[np.random.choice(X.shape[0], K, replace=False), :]
    
    return centroidesIniciais

#%%
# *** Encontrar centroide mais próximo ***

def encontraCentroidesProximos(X, centroides):
    ''' Calcula o centróide mais próximo de cada exemplo'''
    # Inputs:
    # Retorna:
    # Vetor de comprimento n_exemplos com valores dos índices dos centroides
    # n_exemplos = X.shape[0]

    indicesCentroidesProximos = np.zeros(len(X))
    
    for i in range(len(X)):
        #distancia = [ distance.euclidean(X_exemplo, i) for i in centroides]
        
        distancia = [ distance.euclidean(X[i], j) for j in centroides]
        cluster = np.argmin(distancia)
        indicesCentroidesProximos[i] = cluster
            
    return indicesCentroidesProximos

#%%
# *** Cálculo de médias ***
def calcularMediasCentroides(X, idc, K):
    ''' Calcula a média dos valores dos exemplos de cada centroide '''
        
    for i in range(K):
    #pontos = 
        centroides[i] = np.mean(X[idc==i][:], axis=0)
        #print("média com índice = 0", np.mean(X[idx==0][:]))
    #print(centroides)
        
    return np.vstack(centroides)

#%%
# *** Algoritmo K-médias ***

def kMeans(X, K, num_iter, plot_caminho=False):
    # Inicializa aleatoriamente os centroides
    plt.figure()

    centroides = kMeansInicCentroides(X, K)
    plt.scatter(centroides[:, 0], centroides[:, 1], marker='X', 
                        alpha=0.8, c='m', s=150)
    
    # Loop de iteração para aproximação dos grupos aos centroides
    for _ in range(num_iter):
        # A cada iteração, atribui-se inicialmente cada exemplo ao centroide
        # mais próximo
        indicesGrupos = encontraCentroidesProximos(X, centroides)
        
        # Atualiza o valor dos centroide para a média dos exemplos atribuidos
        # a ele.
        #centroides = novosCentroides
        centroidesAntigos = centroides
        centroides = calcularMediasCentroides(X, indicesGrupos, K)
        
        print("São iguais?", np.array_equal(centroidesAntigos, centroides))
        print("Centroides antigos: ")
        print(centroidesAntigos)
        print("Centroides atualizados: ")
        print(centroides)
        
        if plot_caminho:
            # Plotar os centroides e o caminho feito após as atualizações
            for j in range(K):
                plt.arrow(centroidesAntigos[j, 0], centroidesAntigos[j, 1], 
                          centroides[j, 0] - centroidesAntigos[j, 0], 
                          centroides[j, 1] - centroidesAntigos[j, 1])
#            for j in range(K):
#                plt.plot(centroidesAntigos[j, 0], centroides[j, 1], "g--",
#                         alpha=0.7)
        
        plt.scatter(centroides[:, 0], centroides[:, 1], marker='*', c='k', 
                    s=200, alpha = 0.6)
        #plotKmeans(X, centroides, indicesGrupos)
        
        # Condição de parada: se os centroides antigos e novos são iguais, 
        # então não está havendo modificação na atualização.
        if np.array_equal(centroidesAntigos, centroides):
            break
    
    plotKmeans(X, centroides, indicesGrupos)
    #plt.show()
    return centroides, indicesGrupos

#%%
# *** Plot atualização centroides ***

def plotKmeans(X, centroides, indicesGrupos):

    LABEL_COLOR_MAP = {0 : 'r', 1 : 'b'}
    label_color = [LABEL_COLOR_MAP[l] for l in indicesGrupos]
    
    plt.scatter(X[:, 0], X[:, 1], s=6, c=label_color)
    plt.scatter(centroides[:, 0], centroides[:, 1], marker='*', c='k', s=200,
                alpha = 0.6)
    plt.show()

#%%
# *** Aplicação ***

#X = np.random.randint(50, size=(50, 2))
    
# Geração de base de dados para agrupamento
X, y = make_blobs(n_samples=100, random_state=19, centers=3, cluster_std=2)

# 1) Gera centroides aleatoriamente
#centroides = kMeansInicCentroides(X, 2)

# 2) A partir dos centroides, calcula o centroide mais próximo de cada exempl.
#idc = encontraCentroidesProximos(X, centroides)

# Uso do algoritmo geral que retorna o centroide final, e o o cluster a qual
# pertence cada exemplo

centroides, indicesGrupos = kMeans(X, 3, 5, plot_caminho=True)

#plotKmeans(X, centroides, indicesGrupos)