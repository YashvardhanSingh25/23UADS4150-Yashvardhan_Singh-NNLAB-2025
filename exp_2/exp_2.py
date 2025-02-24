# Write A Program to implement a multi-layer perceptron (MLP) network with one hidden layer using numpy in Python. Demonstrate that it can learn the XOR Boolean function.  
import numpy as np

def step_function(x):
    return np.where(x >= 0, 1, 0)

def train_xor_mlp(epochs=10000, learning_rate=0.1):
    # XOR input and output
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Initialize weights and biases
    np.random.seed(1)
    input_neurons = 2
    hidden_neurons = 2
    output_neurons = 1
    
    weights_input_hidden = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
    bias_hidden = np.random.uniform(-1, 1, (1, hidden_neurons))
    weights_hidden_output = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
    bias_output = np.random.uniform(-1, 1, (1, output_neurons))
    
    # Training process
    for _ in range(epochs):
        # Forward pass
        hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_output = step_function(hidden_input)
        final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        final_output = step_function(final_input)
        
        # Backpropagation
        error = y - final_output
        d_output = error
        
        error_hidden = d_output.dot(weights_hidden_output.T)
        d_hidden = error_hidden
        
        # Update weights and biases
        weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
        bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        weights_input_hidden += X.T.dot(d_hidden) * learning_rate
        bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    
    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

def predict_xor_mlp(X, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = step_function(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = step_function(final_input)
    return final_output

# Train the MLP
weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = train_xor_mlp()

# Test the trained MLP
X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_pred = predict_xor_mlp(X_test, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

print("Predictions:", y_pred.flatten())
