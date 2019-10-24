#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tutorial PCA: exemplo variância explicada
"""

#https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

#%% Arindo a base de dados

mnist = fetch_openml('mnist_784', version=1, cache=True)

print(mnist.data.shape)
print()

#%% *** Aplicação do PCA

porcentagem_var_explicada = 0.2

pca = PCA(porcentagem_var_explicada)  # definindo 0.95 de variância

mnist_PCA = pca.fit_transform(mnist.data)

print("número de componentes", pca.n_components_)

aproximacao = pca.inverse_transform(mnist_PCA)

#%% Comparação qualidade de imagens

plt.figure(figsize=(8,4));

indice = 12

# Imagem Original
plt.subplot(1, 2, 1);
plt.imshow(mnist.data[indice].reshape(28,28),
              cmap = plt.cm.gray, interpolation='nearest',                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
              clim=(0, 255));
plt.xlabel('784 componentes', fontsize = 12)
plt.title('Imagem original', fontsize = 16);

# N componentes principais
plt.subplot(1, 2, 2);
plt.imshow(aproximacao[indice].reshape(28, 28),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255));
plt.xlabel('%d componentes'%pca.n_components_, fontsize = 12)
plt.title('%d%% de variância explicada'%(100*porcentagem_var_explicada),
          fontsize = 16);