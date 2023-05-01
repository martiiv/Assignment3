import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rand

#Task 2.1 Is to create a perceptron which can handle binary classification 
def createPerceptron(input,initialWeights, targetValues, learningRate):  
    
    #Defining variables 
    weights = initialWeights
    output = [0, 0, 0, 0]          # Initializing neurons
    datasetSize = len(input)    
    error = []
    
    
    # ? Starting algorithm _________________________________________________________________
    for i in range(0, datasetSize): # We iterate through the dataset'
        
        variables = input.iloc[i]   # We get the variables for each row in the dataset
        targetValue = targetValues.iloc[i] # We get the target value for each row in the dataset
                
        # First we do forward propagation to get the output of the perceptron
        for i in range(0,4):
            output[i] = activationFunction(variables, weights) # We do forward propagation for each neuron and get the output
            weights = updateWeights(weights, learningRate , output, targetValue, variables)    
            
    
        
    
        
# Function takes in input and weights and returns the output of one neuron by multiplying the input with asscoiated weights and adding the bias 
# The output is then passed through the sigmoid function
def activationFunction(input, weights): 
    processedInput =  []
    for i in range(0,4):
        newValue = input[i]*weights[i]
        processedInput.append(newValue)
    return sigmoidFunction(sum(processedInput))        

# Sigmoid function takes in the attribute and returns the sigmoid of the attribute
def sigmoidFunction(attr):
    return (1/(1+np.exp(-attr))) # Sigmoid function

def DerivedSigmoidFunction(attr):
    return (sigmoidFunction(attr)*(1-sigmoidFunction(attr))) # Derivative of the sigmoid function


#Delta learning rule taken from tutorial 6
def deltaLearningRule(targetValues, outputValues): 
    values = []                                                         # Initializing values for each neuron                                              
    for i in range(0,4):
        values.append((1/2)*((targetValues[i]-outputValues[i])**2))     # Calculating the error for each neuron
    error = sum(values)                                                 # Summing the error for each neuron
    return error

def updateWeights(weights, learningRate , output, targetValue, input):
    vectorWeights = np.array(weights)
    vectorInput = np.array(input)
    
    for i in range(weights):
        weights[i] = weights[i] - learningRate*-(targetValue-output)*DerivedSigmoidFunction(np.dot(vectorInput, vectorWeights))*input[i]
    return weights   
        
# ! Defining vairalbes and initializing various components in the algorithm  
initialWeights = []
for i in range(0,4):           # Initializing weights to random values between 0 and 1
    x = rand.random()
    initialWeights.append(x)

## Importing the dataset ################################################################################################
df = pd.DataFrame(pd.read_csv('irisBinary.csv', sep=','))
df.replace(('Iris-setosa', 'Iris-versicolor'), (0, 1), inplace=True)        # Labeling and making the data binary
    
targetValues = df["Iris Class"]
dataset = df.drop(["Iris Class"], axis=1)                                   # Dropping the target value from the dataset
#########################################################################################################################


output = createPerceptron(dataset,initialWeights, targetValues)

