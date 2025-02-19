# README: Network for MNIST Classification

## What is This Program?

This Python program uses **TensorFlow** to train and evaluate a **three-layer neural network** for classifying handwritten digits from the **MNIST dataset**.



## How Does It Work?

1. **Loads the MNIST dataset**: This dataset contains 60,000 training images and 10,000 test images of handwritten digits (0-9).
2. **Prepares the data**:
   - Scales pixel values to be between 0 and 1 (normalization).
   - Flattens the 28x28 images into a single 784-length array.
3. **Defines the neural network**:
   - **Input layer**: 784 neurons (for each pixel of the image).
   - **Hidden layer 1**: 128 neurons with ReLU activation.
   - **Hidden layer 2**: 64 neurons with ReLU activation.
   - **Output layer**: 10 neurons with Softmax activation (one for each digit).
4. **Compiles and trains the model**:
   - Uses **Adam optimizer** and **Sparse Categorical Crossentropy loss**.
   - Trains for **10 epochs**.
5. **Evaluates performance on the test dataset**.
6. **Makes predictions** and displays a few images with predicted labels.



## Required Libraries

- **TensorFlow**: For building and training the neural network.
- **Matplotlib**: For displaying images and predictions.
- **NumPy**: For numerical computations.


## Running the Code

Simply execute the Python script. If TensorFlow is not installed, install it using:

```sh
pip install tensorflow
```


## Expected Output

- The model will print training progress, including loss and accuracy for each epoch.
- Final test accuracy will be displayed, typically around **97-98%**.
- The script will show a few test images along with predicted and actual labels.



## Example Output

```sh
Epoch 1/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.2904 - accuracy: 0.9172 - val_loss: 0.1415 - val_accuracy: 0.9578
...
Epoch 10/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0780 - accuracy: 0.9760 - val_loss: 0.0833 - val_accuracy: 0.9742
Test Accuracy: 0.9742
```

Additionally, the script will display 5 test images along with their predicted labels.

