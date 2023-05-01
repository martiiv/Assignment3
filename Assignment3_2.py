import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.DataFrame(pd.read_csv('irisBinary.csv', sep=',', header=None))
print(df.head)

#Task 2.1 Is to create a perceptron which can handle binary classification
def createPerceptron(neuron1, neuron2, neuron3, neuron4):
    # Initializing random weights between 0 and 1 for each neuron 
    weights = np.random.rand(4,1)
    bias = 0                        # Initializing bias to 0 since i have no clue wether or not i will need it 
    
    #We will have 3 layers in our perceptron
    results = []
    for i in range(0,3):
        print(weights[i])
        #Each loop iteration is one layer in the perceptron
        
# Function takes in input and weights and returns the output of one neuron by multiplying the input with asscoiated weights and adding the bias 
# The output is then passed through the sigmoid function
def activationFunction(input, weights, bias): 
    processedInput = []
    for i in range(weights):
        processedInput.append(input[i]*weights[i]+bias)
    return sigmoidFunction(sum(processedInput))        

def sigmoidFunction(attr):
    return (1/(1+np.exp(-attr))) # Sigmoid function

createPerceptron(1,2,3,4)
    

