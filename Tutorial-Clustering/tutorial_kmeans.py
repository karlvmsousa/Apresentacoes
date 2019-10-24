#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tutorial K-means
@author: karlvandesman
"""

# *** Importar bibliotecas ***
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

#%%
# *************************************
# *** Análise Exploratória de Dados ***
# *************************************
from pandas.plotting import scatter_matrix

print(df.head())
print()

print("Dimensões da base:", df.shape)
print()

print(df.info())
print()

print(df.describe())
print()

df.hist(figsize=[10, 10])
plt.show()

paleta_cores = {0: 'green', 1: 'red'}
cores = [paleta_cores[c] for c in df['classe']]

scatter_matrix(df[atributos], figsize=[11, 11], c=cores)
plt.show()

#%%
# *************************
# *** Pré-processamento ***
# *************************

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

minMaxScaler = MinMaxScaler(feature_range=(0, 1))
standardScaler = StandardScaler()   # média 0 e desvio padrão 1

Xescalonado = minMaxScaler.fit_transform(X)
Xpadronizado = standardScaler.fit_transform(X)

# Aqui podemos ver a conformidade da transformação
print("Mínimos do X escalonado: ", Xescalonado.min(axis=0))
print("Máximos do X escalonado: ", Xescalonado.max(axis=0))
print()

print("Médias do X (standardScaler): ", Xpadronizado.mean(axis=0))
print("Desvios do X (standardScaler): ", Xpadronizado.std(axis=0))
print()

#%%
# ***************************
# *** Aplicação do KMeans ***
# ***************************

modelo_default = KMeans()

# Aqui vemos o objeto criado, com seus parâmetros padrões
print("Modelo padrão do KMeans:\n", modelo_default)
print()

modelo = KMeans(init='random', n_init=10, max_iter=100, tol=1e-6, 
                precompute_distances='auto', random_state=2019, 
                n_clusters = 6)

modelo.fit(Xpadronizado)

#print(modelo.labels_)
grupos, contagem = np.unique(modelo.labels_, return_counts=True)

# Distribuição dos exemplos nos grupos
plt.bar(grupos, contagem);
print()

#%%
from mpl_toolkits.mplot3d import Axes3D

# "Treinando" o modelo
modelo.fit(Xpadronizado)

# O método predict pode ser usado no contexto de aprendizado semi-
# -supervisionado. Primeiramente fazendo o agrupamento de uma grande
# quantidade de dados, depois usa os centroides como features para
# um problema de aprendizado supervisionado.

grupos = modelo.predict(Xpadronizado)
print(grupos)

C = modelo.cluster_centers_    #posição (em cada feature) dos centroides

fig = plt.figure(figsize=[5, 5])
ax = Axes3D(fig)
ax.scatter(Xpadronizado[:, 0], Xpadronizado[:, 1], Xpadronizado[:, 2], y)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='k', s=500)
plt.show()

#%%
# ***************************
# *** Escolha do melhor K ***
# ***************************
from sklearn.metrics import silhouette_score

# *** Método "Elbow" ***
wcss = []
maxit = 30

for num_clusters in range(2, maxit):
    modelo_X = KMeans(n_clusters=num_clusters)    
    modelo_X.fit(X)
    
    print(num_clusters, modelo_X.inertia_)
    wcss.append(modelo_X.inertia_)



plt.title('Método Elbow')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.plot(range(2, maxit), wcss)
plt.show()

#%%
# *** Coeficiente de silhueta ***
coef_silhueta = []

for num_clusters in range(2, maxit):
    modelo_X = KMeans(n_clusters=num_clusters)    
    modelo_X.fit(Xpadronizado)
    silhouette_avg = silhouette_score(Xpadronizado, modelo_X.labels_)
    print("Para n_clusters =", num_clusters,
          "Coeficiente médio de silhueta :", silhouette_avg)
    coef_silhueta.append(silhouette_avg)

plt.title('Coeficiente de silhueta')
plt.xlabel('Número de Clusters')
plt.ylabel('Coeficiente de silhueta')
plt.plot(range(2, maxit), coef_silhueta)
plt.show()

#%%
# *** Plot dos dois gráficos ***
fig, ax1 = plt.subplots()

cor1 = 'tab:red'
cor2 = 'tab:blue'

ax1.set_xlabel('Número de Clusters')
ax1.set_ylabel('WCSS', color=cor1)
ax1.plot(range(2, maxit), wcss, color=cor1)
ax1.tick_params(axis='y', labelcolor=cor1)

ax2 = ax1.twinx()
ax2.set_ylabel('Coeficiente de silhueta', color=cor2)
ax2.plot(range(2, maxit), coef_silhueta, color=cor2)
ax2.tick_params(axis='y', labelcolor=cor2)

fig.tight_layout()
plt.show()