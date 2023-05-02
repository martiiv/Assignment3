import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rand
class neuron: 
    def __init__(self):
        self.inputValue = 0                 # Result of signmoid function 
        self.error = 0                       # Error produced from the delta learning rule 
        self.weights = []                    # [0] = input 1 weight
                                    # [1] = input 2 weight
                                    # [2] = input 3 weight
                                    # [3] = input 4 weight
                                    # [4] = output weight
    
        for i in range(0,4):           # Initializing weights to random values between 0 and 1
            x = rand.random()
            self.weights.append(x)

    
    

#Task 2.1 Is to create a perceptron which can handle binary classification 
def createPerceptron(input, targetValues, learningRate, numOfEpoch):  
    neuron1 = neuron()                          # Initializing the neurons
    neuron2 = neuron()                         # Each neuron has 4 weights associated with it 
    neuron3 = neuron()                         # 3 from the input layer and one for the output neuron
    outputNeuron = neuron()                    # The output neuron has 3 weights associated with it
    
    outputNeuron.weights = [rand.random(), rand.random(), rand.random()]
    #Pseudo code for the algorithm: 
    # Define the input: 4 attributes one fore each column in the dataset
    # Each input is fed to each neuron 
    #   input 1 to neuron 1,2 and 3
    #   input 2 to neuron 1,2 and 3
    #   input 3 to neuron 1,2 and 3   
    #!   For each neuron weights 1,2,3 and 4 is associated with the respective inputs 
    #    Feed weights and input to activation function 
    for i in range(numOfEpoch):
        
        variables = input.iloc[i]                   # We take the first row of the dataset
        target = targetValues.iloc[i]               # We take the first row of the target values
        
        neuron1.inputValue = activationFunction(variables, neuron1.weights)                   # Feeding the input to the neurons  
        neuron1.error = deltaLearningRule(target, neuron1.inputValue)                         # Calculating the error for the neuron
        neuron1.weights = updateWeights(neuron1.weights, learningRate, neuron1.inputValue, target, variables) # Updating the weights for the neuron
        
        
        neuron2.inputValue = activationFunction(variables, neuron2.weights)
        neuron2.error = deltaLearningRule(target, neuron2.inputValue)
        neuron2.weights = updateWeights(neuron2.weights, learningRate, neuron2.inputValue, target, variables) # Updating the weights for the neuron
        
        neuron3.inputValue = activationFunction(variables, neuron3.weights)
        neuron3.error = deltaLearningRule(target, neuron3.inputValue)
        neuron3.weights = updateWeights(neuron3.weights, learningRate, neuron3.inputValue, target, variables) # Updating the weights for the neuron
        
        outputNeuron.inputValue = activationFunction([neuron1.inputValue, neuron2.inputValue, neuron3.inputValue], outputNeuron.weights)
        outputNeuron.error = deltaLearningRule(target, outputNeuron.inputValue)
        outputNeuron.weights = updateWeights(outputNeuron.weights, learningRate, outputNeuron.inputValue, target, [neuron1.inputValue, neuron2.inputValue, neuron3.inputValue]) # Updating the weights for the neuron
        
        
        print("Epoch: ", i, "\n")
        print("Output from neural network" , outputNeuron.inputValue, "Target value: ", target, "\n")
        
    print(outputNeuron.inputValue)
        
        
                
        
# Function takes in input and weights and returns the output of one neuron by multiplying the input with asscoiated weights and adding the bias 
# The output is then passed through the sigmoid function
def activationFunction(input, weights): 
    processedInput =  []
    for i in range(len(weights)):                            # For each neuron we look at one feature (For this case Sepal length, sepal width, petal length and petal width)                          
        newValue = input[i]*weights[i]              # Each feature gets multiplied with the weight associated with the neuron
        processedInput.append(newValue)             # We store the value and apply the sigmoid function to it
    return sigmoidFunction(sum(processedInput))        

# Sigmoid function takes in the attribute and returns the sigmoid of the attribute
def sigmoidFunction(attr):
    return (1/(1+np.exp(-attr))) # Sigmoid function

def DerivedSigmoidFunction(attr):
    return (sigmoidFunction(attr)*(1-sigmoidFunction(attr))) # Derivative of the sigmoid function


#Delta learning rule taken from tutorial 6
def deltaLearningRule(targetValue, outputValue): 
    return ((1/2)*((targetValues-outputValue)**2))     # Calculating the error for each neuron

def updateWeights(weights, learningRate , output, targetValue, input):
    vectorWeights = np.array(weights)
    vectorInput = np.array(input)    
    
    newWeights = vectorWeights + learningRate*((targetValue-output)*DerivedSigmoidFunction(vectorInput*weights))*vectorInput
    return newWeights
        

## Importing the dataset ################################################################################################
df = pd.DataFrame(pd.read_csv('irisBinary.csv', sep=','))
df.replace(('Iris-setosa', 'Iris-versicolor'), (0, 1), inplace=True)        # Labeling and making the data binary
    
targetValues = df["Iris Class"]
dataset = df.drop(["Iris Class"], axis=1)                                   # Dropping the target value from the dataset
#########################################################################################################################

learningRate = 0.1
numOfEpoch = 100

output = createPerceptron(dataset, targetValues, learningRate, numOfEpoch)