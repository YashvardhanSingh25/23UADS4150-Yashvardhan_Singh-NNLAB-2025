# README: Implementing a Perceptron Using NumPy in Python

## What is a Perceptron?
A Perceptron is a basic unit of a neural network. It helps a computer learn how to classify things into categories. Think of it as a very simple brain that can make decisions like "yes or no" or "true or false."

### Example:
If you teach a perceptron to recognize apples and oranges based on size and color, it will eventually learn to tell them apart!

---

## How Does a Perceptron Work?
1. **Takes Inputs**: The perceptron receives inputs (like size and color).
2. **Applies Weights**: It multiplies each input by a number called a weight.
3. **Sums Up Everything**: It adds up all the weighted inputs.
4. **Decides Using an Activation Function**: If the sum is big enough, it outputs 1 (yes), otherwise 0 (no).
5. **Learns Over Time**: It adjusts the weights by learning from mistakes.

---

## Code Explanation

### Step 1: Import Required Library
We use NumPy to handle calculations easily.
```python
import numpy as np
```

### Step 2: Define the Perceptron Class
We create a class that stores:
- **Weights** (which the perceptron learns over time)
- **Learning Rate** (how fast it learns)
- **Epochs** (how many times it repeats learning)

```python
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs
```

### Step 3: Activation Function
This function helps the perceptron decide whether to output 1 (yes) or 0 (no).
```python
    def activation_function(self, x):
        return 1 if x >= 0 else 0
```

### Step 4: Prediction Function
This function calculates the weighted sum and applies the activation function.
```python
    def predict(self, x):
        x = np.insert(x, 0, 1)  # Insert bias term
        return self.activation_function(np.dot(self.weights, x))
```

### Step 5: Training Function
This is where the perceptron learns! It adjusts weights based on mistakes.
```python
    def train(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Insert bias term
        for _ in range(self.epochs):
            for i in range(len(y)):
                prediction = self.activation_function(np.dot(self.weights, X[i]))
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]
```

### Step 6: Running the Perceptron on Example Data
We train the perceptron using an **AND logic gate** dataset and check if it learned correctly.
```python
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    perceptron = Perceptron(input_size=2)
    perceptron.train(X, y)
    
    for sample in X:
        print(f"Input: {sample}, Predicted: {perceptron.predict(sample)}")
```

---

## What This Code Does
1. **Creates a Perceptron** with two inputs.
2. **Trains it** using the AND logic gate data.
3. **Tests it** by predicting outputs for all possible inputs.

### Expected Output:
```
Input: [0 0], Predicted: 0
Input: [0 1], Predicted: 0
Input: [1 0], Predicted: 0
Input: [1 1], Predicted: 1
```
This means the perceptron correctly learned the AND gate logic!

---

## Summary
✅ A perceptron is a simple machine learning model.
✅ It learns by adjusting weights over time.
✅ It uses an activation function to decide outputs.
✅ Our example trained a perceptron using an AND logic gate.

This is the foundation for deep learning and more advanced AI models!

Would you like README files for the other two implementations as well? 😊

