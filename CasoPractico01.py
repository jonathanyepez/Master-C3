# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:03:09 2020

@author: Jonathan A. Yepez M.
"""

#Task Description
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

TASK: Crear un modelo para que se pueda estimar el consumo de un vehículo a partir del resto de las variables
"""
#Import the libraries that will be used in this case
import pandas as pd
from pandas.plotting import scatter_matrix #for a specific stage in EDA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation

#Read the data and create a dataframe
df = pd.read_csv("auto.csv")
print("--------------------")
print(df.info()) #a quick view of the dataframe structure
print("--------------------")
print(df.describe()) #a more in-depth description of the information contained in the df
print("--------------------")
print(df.head()) #show the first 5 entries of the dataframe
print("the columns for this dataframe are:")
print(df.columns)

#we check for missing values (NAs)
#print(df.isnull().sum())
if(df.isnull().sum().sum()== 0):
    print("\nThere are NO missing values.\n")
else:
    print("\nThere are",df.isnull().sum().sum(),"missing values in the dataframe!\n")

#EDA => Exploratory Data Analysis
df = df.drop_duplicates() #remove duplicates (if applicable)

#Scatter matrix for the whole data
scatter_matrix(df, figsize = (12, 12), diagonal = 'kde');

plt.figure(figsize=(10, 6)) #size configuration for plotting
sns.distplot(df['mpg'], color='b', hist_kws={'alpha': 0.4}); #we first generate a distribution plot for 'mpg'
#Se nota una tendencia a un consumo entre 20 y 30 mpgs dentro del dataset.

df_cont = df.select_dtypes(include = ['float64']) #we select the continuous variables 
print("The continuous variables for this case are: \n")
print(df_cont.head())

#Analysing the continuous variables -> scatter plots
for i in range(len(df_cont.columns)-1):
    sns.pairplot(data=df_cont, x_vars=df_cont.columns[i], y_vars=['mpg'], height = 5, aspect=2) #scatter plot vars vs 'mpg'

"""
En este caso, notamos una relación inversamente proporcional entre el consumo (mpg) y las
variables displacement, horsepower, y weight. Esto nos indica que a mayor potencia y 
desplazamiento (en términos del motor), el consumo de gasolina será mayor y por ende se
tendrá un menor 'rendimiento' de mpg.
En el caso de la variable acceleration, se nota un grafico de dispersión sin una tendencia
clara. Esto puede deberse a que esta característica varía entre modelos y tipos de carro
casi intependientemente del consumo de gasolina.
"""
        
df_cat = df.select_dtypes(include = ['int64']) #we select the 'categorical' variables.
print("\nThe categorical variables for this case are: \n")
print(df_cat.head())

for i in range(len(df_cat.columns)):
    sns.catplot(x=df_cat.columns[i], y="mpg", data=df, alpha=0.5) #gnerate a catplot
    ax = sns.boxplot(x=df_cat.columns[i], y="mpg", data=df) #add a boxplot on top of the catplot
    plt.setp(ax.artists, alpha=0.6, edgecolor="k")

"""
Tras haber presentado las gráficas para las variables categóricas, se nota que el número 
de cilindros muestra cierta tendencia en términos generales. A grosso modo, se puede asumir
que cuando un vehículo tiene 8 cilindros, el rendimiento en mpg tiende a ser notablemente 
menor a uno que tenga 4 cilindros.
Asi mismo, el origen del vehículo indica que, si bien es cierto no hay una variación extrema, 
se puede asumir que aquellos provenientes del país '3', tienen un mejor consumo de 
gasolina.
En el caso del año de fabricación (model_year), se observa una tendencia general a la mejora
de mpg conforme se avanza en el tiempo. Esto puede deberse a los avances en el área de la
mecánica automotriz  y la implementación de mejores diseños a nivel de componentes mecánicos
como aerodinámicos.
"""

#Implementing the Regression Model
indep_vars = df.iloc[:,:-1] #selecting 'independent variables'
dep_var = df[["mpg"]] #selecting the 'dependent variable'

#separating the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(indep_vars,dep_var)

model = LinearRegression() #constructor that initializes a LinearRegression object

model.fit(x_train, y_train) #fit the model using the training set

pred_train = model.predict(x_train) #prediction based on the training set
pred_test = model.predict(x_test) #prediction based on the test set

#Checking results using R^2 and MSE
print("====================\n")
print('\nR2 en entrenamiento es: ', round(model.score(x_train, y_train),4))
print('MSE en entrenamiento: ', round(mean_squared_error(y_train, pred_train),2)) 
print("-----------------------")
print('R2 en validación es: ', round(model.score(x_test, y_test),4))
print('MSE en validación es: ', round(mean_squared_error(y_test, pred_test),2))

"""
Se ha obtenido resultados aceptables en terminos de precisión. Dado que los valores de MSE
y R^2 en los test y train sets son similares se puede determinar que no se presenta
overfitting. 
Los parametros de la regresión se muestran a continuación 
"""

print("====================\n")
print("Los parametros de la regresion son: ")
print(model.coef_)
print("El termino independiente de la regresión es: ", model.intercept_)