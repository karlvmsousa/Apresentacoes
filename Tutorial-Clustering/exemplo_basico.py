#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tutorial K-means
@author: karlvandesman
"""

# *** Importar bibliotecas ***
import kmeans_basico

# *** Exemplo de aplicação com base aleatória ***

# Geração de base de dados para agrupamento
X, y = make_blobs(n_samples=100, random_state=19, centers=3, cluster_std=2)

centroides, indicesGrupos = kMeans(X, K=3, num_iter=5, detalhado=True) 