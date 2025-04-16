Here's a clear and informative **README** you can include with your project:

---

# 🧠 CNN Classifier for Fashion MNIST – Experimenting with Hyperparameters

## 📋 Overview

This project uses **Keras (TensorFlow)** to build and evaluate a **Convolutional Neural Network (CNN)** on the **Fashion MNIST** dataset. The goal is to explore how different hyperparameters affect model performance, including:

- Filter size
- Regularization
- Batch size
- Optimizer

---

## 🧰 Dataset

The **Fashion MNIST** dataset is a drop-in replacement for the classic MNIST digits dataset, but with images of **clothing items**:
- 60,000 training images
- 10,000 test images
- 10 classes (T-shirt/top, Trouser, Pullover, etc.)
- Grayscale, 28x28 pixels

---

## 🏗️ Model Architecture

The CNN is built using the **Sequential API** in Keras with the following structure:

- 3 × Convolutional layers with ReLU activation (filter size configurable)
- 2 × MaxPooling layers
- Flatten layer
- Dense layer (64 units)
- Output Dense layer (10 units, softmax for classification)

Regularization can be applied via `kernel_regularizer`.

---

## 🧪 Experiments

We demonstrate how different settings affect accuracy and loss:

### 1. 📦 **Filter Size**
```python
train_and_evaluate(filter_size=(3, 3))
```
Tests how the size of the convolutional filters influences feature extraction and accuracy.

### 2. 🛡️ **L2 Regularization**
```python
train_and_evaluate(regularizer=regularizers.l2(0.01))
```
Applies L2 weight regularization to reduce overfitting by penalizing large weights.

### 3. ⚙️ **Batch Size**
```python
train_and_evaluate(batch_size=64)
```
Compares the effect of a larger batch size on model convergence and stability.

### 4. ⚡ **Optimizer**
```python
train_and_evaluate(optimizer='sgd')
```
Replaces the default Adam optimizer with SGD to evaluate its impact on training speed and performance.

---

## 📈 Visualizations

For each experiment, we generate:
- Accuracy plots (training vs. validation)
- Loss plots (training vs. validation)

This helps visualize the training dynamics and identify overfitting or underfitting.

---

## 🚀 How to Run

1. Make sure you have **TensorFlow 2.x** installed:
```bash
pip install tensorflow
```

2. Run the script:
```bash
python fashion_mnist_experiments.py
```

---

## ✅ Dependencies

- `tensorflow`
- `matplotlib`

---

## 📚 Conclusion

This project helps understand how simple CNNs behave under different training conditions, making it an excellent experiment for beginners and intermediate ML practitioners.

You can expand it by:
- Adding dropout
- Using different datasets (e.g., CIFAR-10)
- Testing deeper networks

---

Let me know if you want this in a `.md` file or rendered in a GitHub-style README! 😡