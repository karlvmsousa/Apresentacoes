#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tutorial K-means
@author: karlvandesman
"""

# *** Importar bibliotecas ***
from sklearn.cluster import KMeans

modelo_default = KMeans(verbose=True)

print(modelo_default)

modelo = KMeans(init='random', n_init=1, max_iter=10, tol=1e-6, 
                precompute_distances='auto')

