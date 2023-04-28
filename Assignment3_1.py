import pandas as pd
#This file will contain the answers to the questions 1.1, 1.2 and 1.3 for Assignment 3 in IMT4133
# @author: Martin Iversen 
# @date: 28.04.2023

#Tas 1.2 Implement K-Means clustering algorithm to perform clusterin on the unlabeled data set unlabeledData.txt
def clusterUnlabeled ():
    samples = 350               # We define known variables
    numAttributes = 90
    
    #First we do some preprocessing and structuring for the unlabeled data
    txt = pd.read_csv('unlabeledData.txt', sep=' ', header=None) #Read the data from the file we separate by space
    txt.to_csv('unlabeledData.csv', index=False, header=None)    #Save the data to a csv file
    
    df = pd.read_csv('unlabeledData.csv')           #Read the data from the csv file
    
    print(df.head())                                            #Print the first 5 rows of the data
    
clusterUnlabeled()
