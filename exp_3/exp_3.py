# Write a Program to evaluate the performance of perceptron with linear and sigmoid activation functions for a regression and binary classification problem respectively. 


import numpy as np
import pandas as pd

class PerceptronGD:
    def __init__(self, input_size, learning_rate=0.01, epochs=100, activation='linear'):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation
    
    def activation_function(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
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
    # Regression problem (Linear Activation)
    regression_data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 3, 4, 5, 6],
        'y': [2.2, 2.8, 3.6, 4.5, 5.1]
    })
    X_reg = regression_data[['x1', 'x2']].values
    y_reg = regression_data['y'].values
    
    perceptron_linear = PerceptronGD(input_size=2, activation='linear')
    perceptron_linear.train(X_reg, y_reg)
    
    print("Linear Activation (Regression):")
    for sample in X_reg:
        print(f"Input: {sample}, Predicted: {perceptron_linear.predict(sample)}")
    
    # Binary classification problem (Sigmoid Activation)
    classification_data = pd.DataFrame({
        'x1': [0, 0, 1, 1],
        'x2': [0, 1, 0, 1],
        'y': [0, 0, 0, 1]
    })
    X_cls = classification_data[['x1', 'x2']].values
    y_cls = classification_data['y'].values
    
    perceptron_sigmoid = PerceptronGD(input_size=2, activation='sigmoid')
    perceptron_sigmoid.train(X_cls, y_cls)
    
    print("\nSigmoid Activation (Binary Classification):")
    for sample in X_cls:
        print(f"Input: {sample}, Predicted: {perceptron_sigmoid.predict(sample)}")
