 import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.randn(input_size)  # Initialize weights randomly
        self.bias = np.random.randn()  # Initialize bias randomly
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return 1 if x >= 0 else 0  # Step function

    def predict(self, x):
        linear_output = np.dot(self.weights, x) + self.bias  # wx + b
        return self.activation_function(linear_output)

    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(y)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]  # Update weights
                self.bias += self.learning_rate * error  # Update bias

# Example usage
if __name__ == "__main__":
    # Training data for AND gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # Expected output for AND gate

    perceptron = Perceptron(input_size=2)
    perceptron.train(X, y)

    # Testing the trained perceptron
    print("Testing Perceptron on AND gate:")
    for sample in X:
        print(f"Input: {sample}, Predicted Output: {perceptron.predict(sample)}")
