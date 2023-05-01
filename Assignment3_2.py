import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rand

#Task 2.1 Is to create a perceptron which can handle binary classification 
def createPerceptron(input, epochs, learningRate):  
    # ! Defining vairalbes and initializing various components in the algorithm  
    weights = []
    for i in range(0,4):           # Initializing weights to random values between 0 and 1
        x = rand.random()
        weights.append(x)
    
    variables = []                  # Variables will be the neuron values stored up 
    neurons = [0, 0, 0, 0]          # Initializing neurons
    results = []                    # Results is a list which will contain the results of each epoch
    
    mse = 0                         # Mean squared error will update after each epoch
    bias = 0                        # Initializing bias to 0 since i have no clue wether or not i will need it 
    
    # ? Starting algorithm _________________________________________________________________
    
    variables = input              # Initializing variables to the input for the first iteration
    
    # First we do forward propagation to get the output of the perceptron
    for i in range(0,4):
        neurons[i] = forwardPropagation(variables, neurons[i], weights, bias) # We do forward propagation for each neuron and get the output
    print(neurons)
    
    weights.clear()                 # Clearing the weights list since we need new weights for the final iteration
    for i in range(0,4):           # Initializing weights again since we need new ones for the final iteration
        x = rand.random()
        weights.append(x)
    
    #Since we currently only have one layer we will funnel all neurons into one neuron which will give us the output
    output = 0
    results.append(forwardPropagation(neurons, output, weights, bias)) # We do forward propagation for the final neuron and get the output

    return results

    
    # Then we do back propagation to update the weights and bias
    #backPropagation(input, weights, bias)
        
def forwardPropagation(input, neuron,  weights, bias):
    # We will use the activation function to get the output of the perceptron for each neuron
    # For our case we have four neurons since we have four attributes, sepal length, sepal width, petal length and petal width
    # Additionally all neurons will have a connection to each input
    neuron = (activationFunction(input, weights, bias)) 
    return neuron
    
        
#def backPropagation(input, weights, bias):
    
        
# Function takes in input and weights and returns the output of one neuron by multiplying the input with asscoiated weights and adding the bias 
# The output is then passed through the sigmoid function
def activationFunction(input, weights, bias): 
    processedInput =  []
    for i in range(0,4):
        newValue = input[i]*weights[i]
        processedInput.append(newValue)
    return sigmoidFunction(sum(processedInput)+bias)        

# Sigmoid function takes in the attribute and returns the sigmoid of the attribute
def sigmoidFunction(attr):
    return (1/(1+np.exp(-attr))) # Sigmoid function

# Simple MSE Function which takes in the observed value, predicted value and number of observations
def meanSquaredError(observedVal, perdictedVal, observations):
    return (1/observations)*sum((observedVal-perdictedVal)**2)

#* ###################################################################################################################

# Importing the dataset
df = pd.DataFrame(pd.read_csv('irisBinary.csv', sep=',', header=None))
df.replace(('Iris-setosa', 'Iris-versicolor'), (0, 1), inplace=True)        # Labeling and making the data binary
print(df.head())


testInput = [5.1,3.5,1.4,0.2]
output = createPerceptron(testInput, 0, 1)
print(output)

