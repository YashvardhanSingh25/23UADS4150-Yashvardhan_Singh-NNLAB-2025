# Write a Program to implement a Perceptron using numpy in Python. 
import numpy as np

class Perceptron:
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
            for i in range(len(y)):
                prediction = self.activation_function(np.dot(self.weights, X[i]))
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]

# Example usage
if __name__ == "__main__":
    # Training data (AND logic gate)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    perceptron = Perceptron(input_size=2)
    perceptron.train(X, y)
    
    # Testing
    for sample in X:
        print(f"Input: {sample}, Predicted: {perceptron.predict(sample)}")
