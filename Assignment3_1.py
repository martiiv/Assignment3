import pandas as pd
import matplotlib.pyplot as plt # We use pyplot to visualize the data
from sklearn.cluster import KMeans # We use KMeans to cluster the data
from sklearn.metrics import silhouette_score # We use silhouette_score to evaluate the clustering
   
#This file will contain the answers to the questions 1.1, 1.2 and 1.3 for Assignment 3 in IMT4133
# @author: Martin Iversen 
# @date: 28.04.2023

#Tas 1.2 Implement K-Means clustering algorithm to perform clusterin on the unlabeled data set unlabeledData.txt
def clusterUnlabeled ():
    samples = 350               # We define known variables
    numAttributes = 90
    ks = []
    wcssm = []
    clusters = []
    
    #First we do some preprocessing and structuring for the unlabeled data
    txt = pd.read_csv('unlabeledData.txt', sep=' ', header=None) #Read the data from the file we separate by space
    txt.to_csv('unlabeledData.csv', index=False, header=None)    #Save the data to a csv file
    
    df = pd.read_csv('unlabeledData.csv')           #Read the data from the csv file
    
    print(df.max())                                   # We check for abnormalities in the data
    print(df.min())
    print(df.mean())                                    
    print("Since the min max and mean all fall between 0 and 1 we wont normalize the data")
    
    #We implemetn k means clustering using the matplotlib library
    for i in range(2,15):                    # For each datapoint we will do the clustering wit k=i clusters
        algorithm = KMeans(         # We define the algorithm
        init="k-means++",              # We initialize the clusters randomly
        n_clusters=i,               # The number of clusters increase for each loop run
        n_init=10,                  # We initialize ten runs for each foor  loop run
        max_iter=100,               # We set the maximimum number of iterations to 100
        random_state=42)

        algorithm.fit(df)           # We fit the algorithm to the data
        ks.append(algorithm.n_iter_)
        wcssm.append(algorithm.inertia_)
        clusters.append(algorithm.labels_)
    
    #After fitting the algorithm we will evaluate the clustering using various metrics 
    #To evaluate the clustering we will plot a graph
    for i in range(0,13):
        print(clusters[i])
        
    plt.plot(range(2,15), wcssm)
    plt.xlabel("K")
    plt.ylabel("WCSS")
    plt.show()
        
clusterUnlabeled()