# 2. Write a Program to implement the Gradient Descent algorithm for perceptron learning using numpy and Pandas.

import numpy as np
import pandas as pd

class PerceptronGD:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def activation_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        x = np.insert(x, 0, 1)  # Insert bias term
        return self.activation_function(np.dot(self.weights, x))
    
    def train(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Insert bias term
        
        for _ in range(self.epochs):
            gradient = np.zeros_like(self.weights)
            for i in range(len(y)):
                prediction = self.activation_function(np.dot(self.weights, X[i]))
                error = y[i] - prediction
                gradient += error * X[i]
            self.weights += self.learning_rate * gradient / len(y)

# Example usage
if __name__ == "__main__":
    # Training data (AND logic gate)
    data = pd.DataFrame({
        'x1': [0, 0, 1, 1],
        'x2': [0, 1, 0, 1],
        'y': [0, 0, 0, 1]
    })
    X = data[['x1', 'x2']].values
    y = data['y'].values
    
    perceptron = PerceptronGD(input_size=2)
    perceptron.train(X, y)
    
    # Testing
    for sample in X:
        print(f"Input: {sample}, Predicted: {perceptron.predict(sample)}")
