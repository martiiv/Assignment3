import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rand
from sklearn.model_selection import KFold # import KFold


# Class for neurons, used to store weights associated with the neuron
class neuron: 
    def __init__(self):
        self.inputValue = 0                 # Result of signmoid function 
        self.weights = []                    # [0] = input 1 weight
                                    # [1] = input 2 weight
                                    # [2] = input 3 weight
                                    # [3] = input 4 weight
                                    # [4] = output weight
    
        for i in range(0,4):           # Initializing weights to random values between 0 and 1
            x = rand.random()
            self.weights.append(x)

neuron1 = neuron()                          # Initializing the neurons
neuron2 = neuron()                         # Each neuron has 4 weights associated with it 
neuron3 = neuron()                         # 3 from the input layer and one for the output neuron
outputNeuron = neuron()                    # The output neuron has 3 weights associated with it
    

# ! Code explanation and HERE!!!!
# ? Since the code is somewhat complex I will try to explain what is happening
# * The algorithm starts by initializing the four neurons we use as well as the associated weights
# * Each neuron has 4 weights associated with it (all corresponding to one attribute in the dataset)
# * The output neuron has 3 weights associated with it (corresponding to the 3 neurons in the hidden layer)

# ? For each epoch we will iterate through the entire dataset
#   We pass the input of one row in the dataset to the three neurons 
#       Each neuron uses the sigmoid activation function to calculate the output of the neuron
#       The weights are then updated using the delta learning rule
#  ! After each neuron in the hidden layer has been updated we pass the output of the three neurons to the output neuron
#       The output neuron uses the sigmoid activation function to calculate the output 
#       The weights are then updated using the delta learning rule

# This process is repeated for each epoch where each epoch is one iteration through the entire dataset
# We store the mean square error for each epoch in a list and plot it after the algorithm has finished


#Task 2.1 Is to create a perceptron which can handle binary classification 
def trainModel(input, targetValues, learningRate, numOfEpoch):  
    neuron1 = neuron()                          # Initializing the neurons
    neuron2 = neuron()                         # Each neuron has 4 weights associated with it 
    neuron3 = neuron()                         # 3 from the input layer and one for the output neuron
    outputNeuron = neuron()                    # The output neuron has 3 weights associated with it
    
    mse = []                                    # List for mean squared error
    
    outputNeuron.weights = [rand.random(), rand.random(), rand.random()] # Since output neuron has 3 inputs we need 3 weights instead of 4
    print(outputNeuron.weights)
    for j in range(numOfEpoch):
        for i in range(0, len(targetValues)):
            
            variables = input.iloc[i]                   # We take the first row of the dataset
            target = targetValues.iloc[i]               # We take the first row of the target values
        
            neuron1.inputValue = activationFunction(variables, neuron1.weights)                   # Feeding the input to the neurons  
            neuron1.weights = updateWeights(neuron1.weights, learningRate, neuron1.inputValue, target, variables) # Updating the weights for the neuron
        
        
            neuron2.inputValue = activationFunction(variables, neuron2.weights)
            neuron2.weights = updateWeights(neuron2.weights, learningRate, neuron2.inputValue, target, variables) # Updating the weights for the neuron
        
            neuron3.inputValue = activationFunction(variables, neuron3.weights)
            neuron3.weights = updateWeights(neuron3.weights, learningRate, neuron3.inputValue, target, variables) # Updating the weights for the neuron
            outputNeuron.inputValue = activationFunction([neuron1.inputValue, neuron2.inputValue, neuron3.inputValue], outputNeuron.weights)
            
            outputNeuron.weights = updateWeights(outputNeuron.weights, learningRate, outputNeuron.inputValue, target, [neuron1.inputValue, neuron2.inputValue, neuron3.inputValue]) # Updating the weights for the neuron
        
        mse.append(meanSquaredError(target, outputNeuron.inputValue, numOfEpoch)) 
           

    plt.plot(range(0,numOfEpoch), mse)
    plt.xlabel("Epoch")
    plt.ylabel("Mean square error")
    plt.show()
    return outputNeuron, neuron1, neuron2, neuron3

def predictWithNN(x, y):
    for i in range(0, len(x)):
        neuron1.inputValue = activationFunction(x.iloc[i], neuron1.weights)
        neuron2.inputValue = activationFunction(x.iloc[i], neuron2.weights)
        neuron3.inputValue = activationFunction(x.iloc[i], neuron3.weights)
        outputNeuron.inputValue = activationFunction([neuron1.inputValue, neuron2.inputValue, neuron3.inputValue], outputNeuron.weights)
        
        if outputNeuron.inputValue >= 0.5:
            print("Guess: Iris-versicolor actual value: ", y.iloc[i])
            return 1
        elif outputNeuron.inputValue < 0.5:
            print("Guess: Iris-setosa actual value: ", y.iloc[i])
            return 0
        
# Function takes in input and weights and returns the output of one neuron by multiplying the input with asscoiated weights and adding the bias 
# The output is then passed through the sigmoid function
def activationFunction(input, weights): 
    processedInput =  []
    for i in range(len(input)):                            # For each neuron we look at one feature (For this case Sepal length, sepal width, petal length and petal width)                          
        newValue = input[i]*weights[i]              # Each feature gets multiplied with the weight associated with the neuron
        processedInput.append(newValue)             # We store the value and apply the sigmoid function to it
    return sigmoidFunction(sum(processedInput))        

# Sigmoid function takes in the attribute and returns the sigmoid of the attribute
def sigmoidFunction(attr):
    return (1/(1+np.exp(-attr))) # Sigmoid function

def DerivedSigmoidFunction(attr):
    return (sigmoidFunction(attr)*(1-sigmoidFunction(attr))) # Derivative of the sigmoid function

# Simple MSE Function which takes in the observed value, predicted value and number of observations
def meanSquaredError(observedVal, perdictedVal, observations):
    return (1/observations)*(observedVal-perdictedVal)**2

# Weight update function which takes in the weights, learning rate, output, target value and input 
def updateWeights(weights, learningRate , output, targetValue, input): # Function is inspired by slides in turorial 5 and 6
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

# Task 2.2 is to perform 5-fold cross validation on the dataset 
# I have used an online tutorial to implement the sklearn cross validation function: https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6 
# Last visit 02.05.2023
kfold = KFold(n_splits=10) # Creating the kfold object
kf = kfold.split(dataset, targetValues) # Splitting the dataset into 5 folds
    
learningRate = 0.01
numOfEpoch = 100

output = trainModel(dataset, targetValues, learningRate, numOfEpoch)
test = predictWithNN(dataset, targetValues)

#for i, (train_index, test_index) in enumerate(kf):
    #print(f"Fold {i}:")
    #print(f"  Train: index={train_index}")
    #print(f"  Test:  index={test_index}")



#! Something wrong with output neurons weights probably line 74