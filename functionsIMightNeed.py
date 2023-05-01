# Simple MSE Function which takes in the observed value, predicted value and number of observations
def meanSquaredError(observedVal, perdictedVal, observations):
    return (1/observations)*sum((observedVal-perdictedVal)**2)