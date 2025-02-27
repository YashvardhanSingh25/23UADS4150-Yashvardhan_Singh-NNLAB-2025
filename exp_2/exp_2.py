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
    
    best_loss = float('inf')
    best_weights_input_hidden = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
    best_bias_hidden = np.random.uniform(-1, 1, (1, hidden_neurons))
    best_weights_hidden_output = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
    best_bias_output = np.random.uniform(-1, 1, (1, output_neurons))
    
    # Random search training
    for _ in range(epochs):
        weights_input_hidden = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
        bias_hidden = np.random.uniform(-1, 1, (1, hidden_neurons))
        weights_hidden_output = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
        bias_output = np.random.uniform(-1, 1, (1, output_neurons))
        
        # Forward pass
        hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_output = step_function(hidden_input)
        final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        final_output = step_function(final_input)
        
        # Compute loss
        loss = np.mean((y - final_output) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_weights_input_hidden = weights_input_hidden
            best_bias_hidden = bias_hidden
            best_weights_hidden_output = weights_hidden_output
            best_bias_output = bias_output
    
    print(f'Best Loss: {best_loss}')
    return best_weights_input_hidden, best_bias_hidden, best_weights_hidden_output, best_bias_output

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
