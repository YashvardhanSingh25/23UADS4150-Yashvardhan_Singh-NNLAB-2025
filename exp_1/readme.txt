# Perceptron Implementation in Python

## Overview
This project implements a **Perceptron** using Python and NumPy. The perceptron is a fundamental unit of an artificial neural network that can solve simple binary classification tasks. In this implementation, the perceptron is trained to learn the **AND logic gate** and then tested for accuracy.

## Formula Used
The perceptron follows the equation:

\[ y = f(w \cdot x + b) \]

where:
- **w** = Weights of the inputs
- **x** = Input values
- **b** = Bias term
- **f** = Activation function (Step function in this case)
- **y** = Predicted output

The step function is defined as:
\[ f(x) = \begin{cases} 1, & x \geq 0 \\ 0, & x < 0 \end{cases} \]

## Code Explanation
### 1. Importing Dependencies
```python
import numpy as np
```
We use **NumPy** for efficient numerical computations.

### 2. Creating the Perceptron Class
```python
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.randn(input_size)  # Initialize random weights
        self.bias = np.random.randn()  # Initialize random bias
        self.learning_rate = learning_rate
        self.epochs = epochs
```
- `input_size`: Number of input features (2 for an AND gate)
- `learning_rate`: Controls weight adjustments
- `epochs`: Number of training iterations
- `weights`: Randomly initialized weight values
- `bias`: Randomly initialized bias value

### 3. Activation Function
```python
    def activation_function(self, x):
        return 1 if x >= 0 else 0  # Step function
```
The **step function** determines whether the perceptron outputs **1** or **0** based on the weighted sum.

### 4. Prediction Function
```python
    def predict(self, x):
        linear_output = np.dot(self.weights, x) + self.bias  # Compute wx + b
        return self.activation_function(linear_output)
```
- Computes the weighted sum **(wx + b)**
- Applies the activation function to generate a prediction

### 5. Training Function
```python
    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(y)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]  # Adjust weights
                self.bias += self.learning_rate * error  # Adjust bias
```
- Iterates through training data for `epochs` iterations
- Computes prediction and error (difference from actual output)
- Updates weights and bias using **error correction**

### 6. Training the Perceptron on AND Gate
```python
if __name__ == "__main__":
    # Training data for AND gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # Expected output for AND gate

    perceptron = Perceptron(input_size=2)
    perceptron.train(X, y)
```
- Defines input (`X`) and expected output (`y`) for the AND logic gate
- Trains the perceptron using `train()` function

### 7. Testing the Perceptron
```python
    print("Testing Perceptron on AND gate:")
    for sample in X:
        print(f"Input: {sample}, Predicted Output: {perceptron.predict(sample)}")
```
- Tests the trained model on all possible inputs
- Prints the predicted output

## Expected Output
After training, the perceptron should correctly classify the AND logic gate:
```
Testing Perceptron on AND gate:
Input: [0 0], Predicted Output: 0
Input: [0 1], Predicted Output: 0
Input: [1 0], Predicted Output: 0
Input: [1 1], Predicted Output: 1
```

## How to Run the Code
1. Install Python (if not installed)
2. Save the code as `perceptron.py`
3. Run the script:
```bash
python perceptron.py
```

## Modifying for Other Logic Gates
To train the perceptron for other logic gates, modify the `y` array:

- **OR Gate:** `y = np.array([0, 1, 1, 1])`
- **NAND Gate:** `y = np.array([1, 1, 1, 0])`
- **NOR Gate:** `y = np.array([1, 0, 0, 0])`

## Conclusion
This implementation demonstrates how a **single-layer perceptron** can solve linearly separable problems like the AND gate. However, it cannot solve problems like XOR, which require a **multi-layer perceptron**.

Feel free to experiment with different logic gates and learning rates!

---
**Author:** Your Name  
**Date:** YYYY-MM-DD

