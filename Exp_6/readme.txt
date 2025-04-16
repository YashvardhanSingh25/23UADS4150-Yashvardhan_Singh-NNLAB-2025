Here's a `README.md` file that explains the code clearly for someone who might be using or studying it for the first time:

---

# 📈 PyTorch RNN for Time Series Forecasting

This project demonstrates how to build and train a simple Recurrent Neural Network (RNN) using PyTorch to predict future values in a time series. It uses a synthetic sine wave as sample data and trains the model to forecast the next value based on a sequence of past values.

---

## 🧠 What This Code Does

- 📊 **Generates a synthetic sine wave** as time series data.
- 🔁 **Prepares input-output pairs** for sequence prediction using a "lookback" window.
- 🔍 **Normalizes data** using MinMaxScaler for improved training stability.
- 🧱 **Defines a simple RNN architecture** with 1 recurrent layer followed by a fully connected layer.
- 🎓 **Trains the model** to minimize Mean Squared Error (MSE).
- 📈 **Evaluates and visualizes predictions** against actual data.

---

## 🗂️ File Structure

```
rnn_time_series.py   # Main training and evaluation script
README.md            # This file
```

---

## 🧾 Requirements

Make sure you have these Python packages installed:

```bash
pip install torch numpy matplotlib scikit-learn
```

---

## 🛠️ How It Works

### 🔹 1. Generate the Data
```python
data = generate_sine_wave(1000)
```
Generates a sine wave with 1000 points. This simulates time series data.

---

### 🔹 2. Normalize the Data
```python
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
```
Min-Max scaling is used to normalize values between 0 and 1 for faster convergence.

---

### 🔹 3. Prepare Sequences
```python
lookback = 50
X, y = prepare_data(data, len(data), lookback)
```
Using a `lookback` of 50 means: the model sees 50 past values to predict the next one.

---

### 🔹 4. Split into Train & Test
```python
train_data, test_data = X[:split], X[split:]
```
80% of data is used for training and 20% for testing.

---

### 🔹 5. Convert to PyTorch Tensors
```python
sequences = torch.FloatTensor(sequences).view(-1, lookback, 1)
```
The data is reshaped for RNN input: `(batch_size, sequence_length, input_size)`

---

### 🔹 6. Define the Model
```python
self.rnn = nn.RNN(input_size, hidden_size, ...)
self.fc = nn.Linear(hidden_size, output_size)
```
A simple RNN is followed by a dense layer to produce the prediction.

---

### 🔹 7. Train the Model
```python
loss = criterion(outputs, batch_target.view(-1, 1))
loss.backward()
optimizer.step()
```
Uses Adam optimizer and MSE loss. Trains for 200 epochs by default.

---

### 🔹 8. Evaluate & Plot
```python
plt.plot(true_values, label='True')
plt.plot(predictions, label='Predicted')
```
After de-normalizing, the predicted values are plotted against the ground truth.

---

## 📊 Output Example

You should see a graph comparing the actual sine wave to the predicted values. Over time, the model learns the sinusoidal pattern quite well.

---

## 🧪 Next Steps / Improvements

- Replace synthetic data with real-world time series (e.g., stock prices, temperature).
- Try deeper RNNs or switch to LSTM/GRU for better memory.
- Add early stopping or learning rate scheduling.

---

## 📬 Contact

Feel free to fork or adapt the script for your own experiments! Let me know if you want help modifying it for real-world applications 😡

--- 

Let me know if you'd like this saved to an actual `README.md` file or need a version with Markdown formatting stripped out!