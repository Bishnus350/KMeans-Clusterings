# Unsupervised ML like Clustering, DBSCAN, PCA and etc
#!C:\Users\bishn\Desktop\VS_python
# Importing python modules 
pwd = r'C:/Users/bishn/Desktop/VS_python/'
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# Importing ML modules 
from sklearn import datasets
iris = datasets.load_iris () # loading iris datasets 
#print (iris)
print ("Name or classes in the datasets",iris.target_names) # target_names give us the column name
print (iris.feature_names)
print ("\n Shape is ", iris.data.shape)
print (iris.data[:10])

# Plot data in plane of first two features (sepal lenght and width)
plt.figure (figsize=(6, 5))
x = iris.data[:, 0] # Sepal length
y = iris.data[:, 1] # Sepal width
z = iris.data[:, 2] # Petal length
a = iris.data[:, 3] # Petal width
plt.scatter (x, y, c = iris.target, s = 100, cmap= "plasma", edgecolors='blue')
plt.xlabel ("Sepal length")
#plt.xlim (0, 10) The limiting are not necessary here in this condition
#plt.ylim (0, 6)
plt.ylabel ("Sepal width")
plt.savefig (pwd + "sepal_plot.png")
#plt.show()
# In this figure we can visually see 2 clusters. Let's use ML algorithm and see what they say.
#K-Means Clustering "https://www.analyticsvidhya.com/blog/2021/11/understanding-k-means-clustering-in
# -machine-learningwith-examples/#:~:text=It%20is%20also%20known%20as,is%20as%20small%20as%20possible."
# Importing K-means clustering modules 
from sklearn.cluster import KMeans
# Initialize K-means with 2 clusters
Kcluster = KMeans (n_clusters= 2, init= "k-means++", n_init=50, random_state= 10 )
Kcluster.fit (iris.data)
inertia2 = Kcluster.inertia_
Klabels = Kcluster.labels_ # Get the cluster assignment for each label 
centroid = Kcluster.cluster_centers_ # Centroids for the  determined cluster
# Plotting the plot 
plt.figure(figsize= (6, 5))
plt.scatter (x, y, c = Klabels, s = 100 , cmap= "plasma", edgecolors='yellow' )
plt.scatter (centroid[:, 0], centroid[:, 1], c = [0, 1], marker= '^', s= 200, cmap='jet')
plt.xlabel ("Sepal length")
plt.ylabel ("Sepal width")
plt.savefig (pwd + 'K-means_2clusters.png')
plt.show ()
# We can see there are two centroids in two different clusters
# Try for 3  clusters 
Kcluster3 = KMeans (n_clusters= 3, init= "k-means++", n_init=50, random_state= 10 )
Kcluster3.fit (iris.data)
inertia3 = Kcluster3.inertia_
Klabels3 = Kcluster3.labels_ # Get the cluster assignment for each label 
centroid3 = Kcluster3.cluster_centers_ # Centroids for the  determined cluster
# Plotting the plot 
plt.figure(figsize= (6, 5))
plt.scatter (x, y, c = Klabels3, s = 100 , cmap= "plasma", edgecolors='yellow' )
plt.scatter (centroid3[:, 0], centroid3[:, 1], c = [0, 1, 2], marker= '^', s= 200, cmap='jet')
plt.xlabel ("Sepal length")
plt.ylabel ("Sepal width")
plt.savefig (pwd + 'K-means_3clusters.png')
plt.show ()

# Try for 4  clusters 
Kcluster4 = KMeans (n_clusters= 4, init= "k-means++", n_init=50, random_state= 10 )
Kcluster4.fit (iris.data)
inertia4 = Kcluster4.inertia_
Klabels4 = Kcluster4.labels_ # Get the cluster assignment for each label 
centroid4 = Kcluster4.cluster_centers_ # Centroids for the  determined cluster
# Plotting the plot 
plt.figure(figsize= (6, 5))
plt.scatter (x, y, c = Klabels4, s = 100 , cmap= "plasma", edgecolors='yellow' )
plt.scatter (centroid4[:, 0], centroid4[:, 1], c = [0, 1, 2, 3], marker= '^', s= 200, cmap='plasma')
plt.xlabel ("Sepal length")
plt.ylabel ("Sepal width")
plt.savefig (pwd + 'K-means_4clusters.png')
plt.show ()
# The inertia represents the sum of squared distance between each data point and it's assigned cluster's centroid.
print ('Kmeans inertia for 2 clusters = ', inertia2)
print ('Kmeans inertia for 3 clusters = ', inertia3)
print ('Kmeans inertia for 4 clusters = ', inertia4)
# The value of k is decreasing as we increase the number of clusters.
# Let's calculate the mean, median, minimum and maximim value of the overall data.
print ("Feature\t\t\tmean\tstd\tmin\tmax")
for featnum, feat in enumerate (iris.feature_names):
    print ("{:s} \t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(feat, np.mean(iris.data[:, featnum]),
                                                          np.std(iris.data[:, featnum]),np.min(iris.data[:, featnum]),
                                                          np.max(iris.data[:, featnum])))

print ("\n Petal length has a higher range. Range is the subtraction between maximum and minimum value."
       "\n The value that have range larger have the more consistent features")
# Features with different scales or ranges can have a disproportionate impact on clustering results.
# In this fluctuated data, it is suggested to Normalize the features between mean 0 to std 1. [0, 1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(iris.data)
Kcluster3 = KMeans (n_clusters=3, n_init= 50)
Kcluster3.fit(scaler.transform(iris.data)) # Perform clustering on the scaled data
x = iris.data[:, 0]
y = iris.data[:, 1]
# Plotting the plot 
plt.figure(figsize= (6, 5))
plt.scatter (x, y, c = Klabels3, s = 100 , cmap= "plasma", edgecolors='yellow' )
plt.xlabel ("Sepal length")
plt.ylabel ("Sepal width")
plt.savefig (pwd + 'Standardscaler.png')
plt.show ()

# Now try another Unsupervised learning techniques 
# Desnity Based Spatial Clustering of Applications with Noise (DBSCAN)
# More about DBSCAN can be found in internet. It make a cluster based on Density

# Importing modules 
from sklearn.cluster import DBSCAN
dbs = DBSCAN (eps = 0.6, min_samples= 7)
dbs.fit (scaler.transform (iris.data))
dbs_outliers = dbs.labels_ == -1
plt.figure (figsize= (6, 5))
plt.scatter (iris.data[:, 0], iris.data[:, 1], c = dbs.labels_, s = 100, edgecolors="None", cmap= "viridis")
plt.scatter (iris.data [:, 0][dbs_outliers], iris.data[:, 1][dbs_outliers], s = 50, c = 'r')
plt.xlabel ("Sepal length")
plt.ylabel ("Sepal width")
plt.savefig (pwd + 'DBSCAN.png')
plt.show ()
# Here DBSCAN can only detect 2 clusters in IRIS data. The red dots do not clusters and is labelled as outliers. 