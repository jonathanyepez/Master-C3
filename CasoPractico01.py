# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:03:09 2020

@author: Jonathan A. Yepez M.
"""

#TASK DESCRIPTION

"""
En el archivo auto.csv se encuentran los siguientes datos de diferentes automoviles:
    * Cilindros
    * Cilindrada
    * Potencia
    * Peso
    * Aceleracion
    * Año del coche
    * Origen
    * Consumo (mpg)
    
Las unidades de las características no se encuentran en el sistema internacional.
La variable 'origen' es un código que identifica el país de orígen.

"""

#Crear un modelo para que se pueda estimar el consumo de un vehículo a partir del resto de las variables


#Import libraries that will be used in thsi case

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
sns.set(color_codes=True)


#Read the data and create a dataframe

df = pd.read_csv("auto.csv")

print(df.info()) #a quick view of the dataframe structure
print(df.describe()) #a more in-depth description of the information contained in the df
print(df.head()) #show the first 5 entries of the dataframe
print("the columns for this dataframe are:")
print(df.columns)

df.dtypes


#we cehck for missing values (NAs)
print(df.isnull().sum())
# since there are no missing values, we do not need to worry about imputation

#EDA => Exploratory Data Analysis

df = df.drop_duplicates() #remove duplicates (if applicable)


plt.figure(figsize=(9, 8))
sns.distplot(df['mpg'], color='b', hist_kws={'alpha': 0.4});


df_cont = df.select_dtypes(include = ['float64'])
df_cont.head()

#Ejemplo de scatter matrix
from pandas.plotting import scatter_matrix

scatter_matrix(df, figsize = (12, 12), diagonal = 'kde');
#==================

for i in range(len(df_cont.columns)-1):
    sns.pairplot(data=df_cont,
                x_vars=df_cont.columns[i],
                y_vars=['mpg'])
        
df_cat = df.select_dtypes(include = ['int64'])
df_cat.head()

plt.figure(figsize = (10, 6))
ax = sns.boxplot(x='origin', y='mpg', data=df)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)


for i in range(len(df_cat.columns)):
    sns.catplot(x=df_cat.columns[i], y="mpg", data=df, alpha=0.5)
    ax = sns.boxplot(x=df_cat.columns[i], y="mpg", data=df)
    plt.setp(ax.artists, alpha=0.6, edgecolor="k")

#Implement Regression model

indep_vars = df.iloc[:,:-1] #selecting 'independent variables'
dep_var = df[["mpg"]] #selecting the 'dependent variable'

#separating the data into train and test sets

x_train, x_test, y_train, y_test = train_test_split(indep_vars,dep_var)

model = LinearRegression() #constructor that initializes a LinearRegression object

model.fit(x_train, y_train)

pred_train = model.predict(x_train)
pred_test = model.predict(x_test)

#Checking results using R^2
print('R2 en entrenamiento es: ', model.score(x_train, y_train))
print('R2 en validación es: ', model.score(x_test, y_test))

