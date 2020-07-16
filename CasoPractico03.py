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
from sklearn import preprocessing #for normalizing
from sklearn.cluster import KMeans #to implement K-Means
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt #plotting

#Read the data from crime_data.csv
df_crime = pd.read_csv('crime_data.csv', index_col=0) #create a dataframe
df_crime_orig = df_crime #create a copy of the originally imported dataframe
print("the dataframe has been created")

#we check the general information of the dataset
print(df_crime.describe())
#print(df_crime.head()) #uncomment to show the first elements of the dataframe

df_crime = preprocessing.scale(df_crime)
df_crime = pd.DataFrame(df_crime) #data standardized

errors = [] #initialize an empty list to store calculated errors
sil =[] #initialize an empty list to store silhouette scores

for i in range(1,15): #consider a range from 1 to 15 clusters
    kmeans = KMeans(i, init = "k-means++", random_state = 42) #method for initialization k-means ++ to
                                                              #speed up convergence.
                                                              #Random state -> random # generation for centroid initialization
    kmeans.fit(df_crime) #computing k-means clustering
    errors.append(kmeans.inertia_) #inertia_ -> Sum of squared distances of samples to their closest cluster center
    if(i>=2): #condition to obtain silhouette scores. 
        sil.append(silhouette_score(df_crime,kmeans.fit_predict(df_crime)))

plt.plot(range(1,15), errors)
plt.title("Elbow Method - Error vs Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Error")
plt.grid()
plt.show()

plt.plot(range(2,15), sil)
plt.title("Silhouette Plot")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid()
plt.show()

"""
Tomando en consideración la gráfica de error vs clusters, se puede establecer que el 
número óptimo de clusters es k=4.
De manera similar, en el caso del Silhouette plot, se observa un pico cuando k=4. Si bien
es cierto, este pico se encuentra a un valor menor al caso de k=2. Se puede sustentar 
nuestra decisión de dividir a nuestro dataframe en 4 grupos.
"""

#Obtaining the centroids for K-Means Clustering
kmeans = KMeans (4, init = "k-means++", random_state=42) #after identifying the optimal number of clusters
var_clustered = kmeans.fit_predict(df_crime)
var_clustered += 1 #to start from 1 instead of 0
df_crime_orig["cluster_group"] = var_clustered #add another column (cluster_group) to the original dataframe

#Mean of clusters 1 to 4
kmeans_centroids = pd.DataFrame(round(df_crime_orig.groupby('cluster_group').mean(),1))

#Plotting to improve visualization
plt.figure(figsize=(12,6)) #establish the figure size

# Scatterplot of the crimes commited in the US 4D presented in a 2D plot
plt.scatter(df_crime_orig["Murder"], df_crime_orig["Assault"], label=None, c=df_crime_orig["Rape"],
            cmap='viridis', s=df_crime_orig["UrbanPop"], linewidth=0, alpha=0.75)
plt.grid(alpha=0.2)
plt.xlim(0, 18)
plt.axis(aspect='equal')
plt.xlabel('Murder')
plt.ylabel('Assault')
plt.colorbar(label='Rape')
# Here we create a legend:
for tam in [30, 60, 90]:
    plt.scatter([], [], c='k', alpha=0.3, s=tam, label=str(tam))
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title='Population')
    plt.title('Crimes in the United States');
    
#plotting 2D Scatter plots to show relation between variables and clusters' positions
colores = {1:"#0066CC",2:"#660066",3:"#009900",4:"black"} #we define a color code for our points

#plot of Murder vs Assault (Graphic #1)          
plt.figure(figsize=(12,6))
plt.scatter(kmeans_centroids["Murder"], kmeans_centroids["Assault"], marker="x", color='r')
for k in range(len(kmeans_centroids)):
    plt.text(kmeans_centroids["Murder"].tolist()[k]+0.25, kmeans_centroids["Assault"].tolist()[k]+0.25, s=("C"+str(k+1)))
    #print(k)    
plt.scatter(df_crime_orig.iloc[:,0],df_crime_orig.iloc[:,1], c=[colores[i] for i in var_clustered])
plt.grid(alpha=0.2)
plt.xlim(0,18)
plt.xlabel('Murder')
plt.ylabel('Assault')
plt.title("Murder vs Assault")
#adding a legend
for k in range(len(kmeans_centroids)):
    plt.scatter([],[],c=colores[k+1], alpha=1, label="C"+str(k+1))
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title="Clusters")

#plot of Murder vs UrbanPop (Graphic #2)           
plt.figure(figsize=(12,6))
plt.scatter(kmeans_centroids["Murder"], kmeans_centroids["UrbanPop"], marker="x", color='r')
for k in range(len(kmeans_centroids)):
    plt.text(kmeans_centroids["Murder"].tolist()[k], kmeans_centroids["UrbanPop"].tolist()[k], s=("C"+str(k+1)))
    #print(k)    
plt.scatter(df_crime_orig.iloc[:,0],df_crime_orig.iloc[:,2], c=[colores[i] for i in var_clustered])
plt.grid(alpha=0.2)
plt.xlim(0,18)
plt.xlabel('Murder')
plt.ylabel('UrbanPop')
plt.title("Murder vs UrbanPop")
#adding a legend
for k in range(len(kmeans_centroids)):
    plt.scatter([],[],c=colores[k+1], alpha=1, label="C"+str(k+1))
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title="Clusters")

#plot of Murder vs Rape (Graphic #3)           
plt.figure(figsize=(12,6))
plt.scatter(kmeans_centroids["Murder"], kmeans_centroids["Rape"], marker="x", color='r')
for k in range(len(kmeans_centroids)):
    plt.text(kmeans_centroids["Murder"].tolist()[k]+0.25, kmeans_centroids["Rape"].tolist()[k]+0.25, s=("C"+str(k+1)))
    #print(k)    
plt.scatter(df_crime_orig.iloc[:,0],df_crime_orig.iloc[:,3], c=[colores[i] for i in var_clustered])
plt.grid(alpha=0.2)
plt.xlim(0,18)
plt.xlabel('Murder')
plt.ylabel('Rape')
plt.title("Murder vs Rape")
#adding a legend
for k in range(len(kmeans_centroids)):
    plt.scatter([],[],c=colores[k+1], alpha=1, label="C"+str(k+1))
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title="Clusters")

#plot of Assault vs UrbanPop (Graphic #4)           
plt.figure(figsize=(12,6))
plt.scatter(kmeans_centroids["Assault"], kmeans_centroids["UrbanPop"], marker="x", color='r')
for k in range(len(kmeans_centroids)):
    plt.text(kmeans_centroids["Assault"].tolist()[k]+0.25, kmeans_centroids["UrbanPop"].tolist()[k]+0.25, s=("C"+str(k+1)))
    #print(k)    
plt.scatter(df_crime_orig.iloc[:,1],df_crime_orig.iloc[:,2], c=[colores[i] for i in var_clustered])
plt.grid(alpha=0.2)
plt.xlabel('Assault')
plt.ylabel('UrbanPop')
plt.title("Assault vs UrbanPop")
#adding a legend
for k in range(len(kmeans_centroids)):
    plt.scatter([],[],c=colores[k+1], alpha=1, label="C"+str(k+1))
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title="Clusters")
    
#plot of Assault vs Rape (Graphic #5)          
plt.figure(figsize=(12,6))
plt.scatter(kmeans_centroids["Assault"], kmeans_centroids["Rape"], marker="x", color='r')
for k in range(len(kmeans_centroids)):
    plt.text(kmeans_centroids["Assault"].tolist()[k]+0.25, kmeans_centroids["Rape"].tolist()[k]+0.25, s=("C"+str(k+1)))
    #print(k)    
plt.scatter(df_crime_orig.iloc[:,1],df_crime_orig.iloc[:,3], c=[colores[i] for i in var_clustered])
plt.grid(alpha=0.2)
plt.xlabel('Assault')
plt.ylabel('Rape')
plt.title("Assault vs Rape")
#adding a legend
for k in range(len(kmeans_centroids)):
    plt.scatter([],[],c=colores[k+1], alpha=1, label="C"+str(k+1))
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title="Clusters")
    
#plot of UrbanPop vs Rape (Graphic #6)           
plt.figure(figsize=(12,6))
plt.scatter(kmeans_centroids["UrbanPop"], kmeans_centroids["Rape"], marker="x", color='r')
for k in range(len(kmeans_centroids)):
    plt.text(kmeans_centroids["UrbanPop"].tolist()[k]+0.25, kmeans_centroids["Rape"].tolist()[k]+0.25, s=("C"+str(k+1)))
    #print(k)    
plt.scatter(df_crime_orig.iloc[:,2],df_crime_orig.iloc[:,3], c=[colores[i] for i in var_clustered])
plt.grid(alpha=0.2)
plt.xlabel('UrbanPop')
plt.ylabel('Rape')
plt.title("UrbanPop vs Rape")
#adding a legend
for k in range(len(kmeans_centroids)):
    plt.scatter([],[],c=colores[k+1], alpha=1, label="C"+str(k+1))
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title="Clusters")


#Displaying the final results in the console
print("The centroids obtained after running K-Means are:\n")
for j in range(4):
    print("-------------------------------------------------")
    print("For Cluster", str(j+1),":\n")
    print("Murder: {:^10}".format(kmeans_centroids["Murder"].tolist()[j]))
    print("Assault: {:^10}".format(kmeans_centroids["Assault"].tolist()[j]))
    print("UrbanPop: {:^8}".format(kmeans_centroids["UrbanPop"].tolist()[j]))
    print("Rape: {:^15}".format(kmeans_centroids["Rape"].tolist()[j]))
    print("\nThe states corresponding to cluster", str(j+1), "are:\n")
    print(df_crime_orig[df_crime_orig['cluster_group']==j+1])
