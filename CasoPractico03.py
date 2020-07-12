# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 23:11:36 2020

@author: Jonathan A. Yepez M.
"""

"""
En el archivo crime_data.csv se encuentra el número de crímenes por cada 100.000 habitantes
en cada uno de los estados de Estados Unidos, así como el porcentaje de la población que es urbana.
Los crímenes se han agrupado en: asalto, asesinato y violación.

Segmenta este conjunto de datos utilizando k-means y obtén los centroides de cada clúster 
y el listado de los estados en cada uno de los clústeres. 
Para ello, se ha de encontrar el número óptimo de clúster en el que se divide el conjunto de datos.

"""
import pandas as pd #data science and information processing
#import numpy as np #algebraic operations
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#import seaborn as sns

df_crime = pd.read_csv("crime_data.csv", index_col=0)

#Hallar el numero optimo de clusters para el conjunto de datos

#we check for a general description of the dataset
print("the dataframe has been created")

print(df_crime.describe())

print(df_crime.head())

df_crime_orig = df_crime

df_crime = preprocessing.scale(df_crime)
df_crime = pd.DataFrame(df_crime) #data normalized

errors = [] 

for i in range(1,15):
    kmeans = KMeans(i, init = "k-means++", random_state = 42)
    kmeans.fit(df_crime)
    errors.append(kmeans.inertia_)

plt.plot(range(1,15), errors)
plt.title("Scree Plot - Error vs Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Error")
plt.grid()
plt.show()
    
kmeans = KMeans (4, init = "k-means++", random_state=42)
y_fitted = kmeans.fit_predict(df_crime)
y_fitted
y_fitted += 1 #to start from 1 instead of 0

df_crime["cluster_group"] = y_fitted

#plt.figure(figsize=(12,6))
#sns.scatterplot(x)
