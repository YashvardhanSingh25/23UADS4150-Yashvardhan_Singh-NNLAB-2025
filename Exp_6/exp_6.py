# Q6.Write a Program to train and evaluate a Recurrent Neural Network using PyTorch Library to predict the next value in a sample time series dataset.
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Function to generate a synthetic sine wave
def generate_sine_wave(sequence_length=1000):
    x = np.linspace(0, 50, sequence_length)  # Generate 1000 evenly spaced values from 0 to 50
    y = np.sin(x)  # Apply sine function
    return y

# Function to prepare the time series dataset into input sequences and targets
def prepare_data(data, sequence_length, lookback):
    sequences = []
    targets = []
    
    # Create sliding window sequences of length `lookback`
    for i in range(len(data) - lookback):
        sequences.append(data[i:i + lookback])      # Input sequence
        targets.append(data[i + lookback])          # Next value to predict
    
    return np.array(sequences), np.array(targets)

# Function to convert NumPy arrays into PyTorch tensors and reshape for RNN input
def to_tensor(sequences, targets):
    # Reshape to (batch_size, sequence_length, input_size=1)
    sequences = torch.FloatTensor(sequences).view(-1, sequences.shape[1], 1)
    targets = torch.FloatTensor(targets)
    return sequences, targets

# Define a simple RNN model using nn.RNN
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  # RNN layer
        self.fc = nn.Linear(hidden_size, output_size)  # Fully connected output layer

    def forward(self, x):
        out, _ = self.rnn(x)  # Pass input through RNN
        out = self.fc(out[:, -1, :])  # Take only the last time step's output
        return out

# Train the RNN model
def train_model(model, train_data, target_data, epochs=100, batch_size=64, lr=0.001):
    criterion = nn.MSELoss()  # Loss function: Mean Squared Error
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Optimizer: Adam

    for epoch in range(epochs):
        model.train()
        
        # Create mini-batches
        for i in range(0, len(train_data), batch_size):
            batch_seq = train_data[i:i + batch_size]
            batch_target = target_data[i:i + batch_size]

            optimizer.zero_grad()  # Clear gradients
            outputs = model(batch_seq)  # Forward pass
            loss = criterion(outputs, batch_target.view(-1, 1))  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        if epoch % 10 == 0:  # Print every 10 epochs
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate the trained model on test data
def evaluate_model(model, test_data, true_values):
    model.eval()
    with torch.no_grad():  # No need to track gradients
        predictions = model(test_data)  # Get model predictions
    return predictions.numpy(), true_values  # Convert predictions to NumPy array

# Function to plot actual vs predicted values
def plot_results(true_values, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label='True Values')
    plt.plot(predictions, label='Predictions')
    plt.legend()
    plt.title("RNN Time Series Forecast")
    plt.show()

# Main script to execute the workflow
if __name__ == "__main__":
    # Generate a sine wave of 1000 data points
    data = generate_sine_wave(1000)
    
    # Normalize the data between 0 and 1 using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
    
    # Convert the time series into sequences for supervised learning
    lookback = 50  # Use the past 50 points to predict the next one
    X, y = prepare_data(data, len(data), lookback)
    
    # Train-test split (80% training, 20% testing)
    split_idx = int(len(X) * 0.8)
    train_data, test_data = X[:split_idx], X[split_idx:]
    train_target, test_target = y[:split_idx], y[split_idx:]

    # Convert data into PyTorch tensors
    train_data, train_target = to_tensor(train_data, train_target)
    test_data, test_target = to_tensor(test_data, test_target)
    
    # Initialize the RNN model
    model = RNNModel(input_size=1, hidden_size=64, output_size=1, num_layers=1)
    
    # Train the model
    train_model(model, train_data, train_target, epochs=200, batch_size=64)
    
    # Evaluate model performance on test set
    predictions, true_values = evaluate_model(model, test_data, test_target)

    # Inverse transform predictions and true values back to original scale
    true_values = scaler.inverse_transform(true_values.reshape(-1, 1)).reshape(-1)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1)
    
    # Plot predictions vs ground truth
    plot_results(true_values, predictions)
