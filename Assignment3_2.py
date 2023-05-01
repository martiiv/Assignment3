import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.DataFrame(pd.read_csv('irisBinary.csv', sep=',', header=None))
print(df.head)

#Task 2.1 Is to create a perceptron which can handle binary classification 
def createPerceptron(neuron1, neuron2, neuron3, neuron4, epochs, learningRate):
    # Initializing random weights between 0 and 1 for each neuron 
    weights = np.random.rand(4,1)
    bias = 0                        # Initializing bias to 0 since i have no clue wether or not i will need it 
    results = []                    # Results is a list which will contain the results of each epoch
                                    # For our case we will use batch learning so we will use all training samples in each epoch
    mse = 0                         # Mean squared error will update after each epoch
                                
    # First we do forward propagation to get the output of the perceptron
    
    # Then we do back propagation to update the weights and bias
    
def forwardPropagation(input, weights, bias):
    
def backPropagation(input, weights, bias):
    
        
# Function takes in input and weights and returns the output of one neuron by multiplying the input with asscoiated weights and adding the bias 
# The output is then passed through the sigmoid function
def activationFunction(input, weights, bias): 
    processedInput = []
    for i in range(weights):
        processedInput.append(input[i]*weights[i]+bias)
    return sigmoidFunction(sum(processedInput))        

# Sigmoid function takes in the attribute and returns the sigmoid of the attribute
def sigmoidFunction(attr):
    return (1/(1+np.exp(-attr))) # Sigmoid function

# Simple MSE Function which takes in the observed value, predicted value and number of observations
def meanSquaredError(observedVal, perdictedVal, observations):
    return (1/observations)*sum((observedVal-perdictedVal)**2)
    

createPerceptron(1,2,3,4)
    

