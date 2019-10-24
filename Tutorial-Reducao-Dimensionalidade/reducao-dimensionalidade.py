#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tutorial PCA
@author: karlvandesman
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 9)

#%%
# ********************************
# *** Leitura da base de dados ***
# ********************************

colunas = ['num-gravidez', 'glicose', 'pressao-diast', 'triceps', 
           'res-insulina', 'imc', 'hist-familia', 'idade', 'classe']

df = pd.read_csv("./datasets/pima-indians-diabetes.csv", names=colunas)

X = df.values[:, 0:8]
y = df.values[:, 8]

atributos = colunas[0:8]

standardScaler = StandardScaler()   # média 0 e desvio padrão 1

Xpadrao = standardScaler.fit_transform(X)

pca = PCA(random_state=2019)
pca.fit(Xpadrao)

pca3 = PCA(n_components=3, random_state=2019)
pca3.fit(Xpadrao)

XPCA = pca3.fit_transform(Xpadrao)
print(XPCA)
print()

#%% Variância explicada

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Número de componentes')
plt.ylabel('Variância explicada cumulativa');
plt.show()

#%% Plot 3D componentes principais

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(XPCA[:, 0], XPCA[:, 1], XPCA[:, 2]);
ax.set_title("Conjunto de dados com PCA e normalização");