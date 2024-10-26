# Fashion MNIST Classification with Artificial Neural Network (ANN)

This project implements a multiclass classification model using Keras with a TensorFlow backend to classify items from the Fashion MNIST dataset. The dataset contains 28x28 grayscale images of 10 different fashion categories, and the goal is to achieve high accuracy in classifying these items using an Artificial Neural Network (ANN).

## Dataset
Fashion MNIST consists of 70,000 images, divided as follows:
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Classes**: 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

Each image is 28x28 pixels in grayscale and belongs to one of the 10 classes.

## Project Steps

### 1. **Data Loading and Preprocessing**
   - Load the dataset using `keras.datasets`.
   - Normalize the images by scaling pixel values to be between 0 and 1.
   - Flatten the 28x28 images into 784-dimensional vectors since we're using an ANN.

### 2. **Model Architecture**
   - We use a sequential neural network with three fully connected (dense) layers:
     - **Input Layer**: Accepts a 784-dimensional flattened vector.
     - **Hidden Layers**: Three dense layers with 256, 128, and 64 neurons respectively, each followed by batch normalization and dropout layers to prevent overfitting.
     - **Output Layer**: 10 neurons with softmax activation to output probabilities for each class.
   - Regularization techniques used:
     - **Dropout**: Applied to each hidden layer to reduce overfitting.
     - **Batch Normalization**: Ensures faster and more stable training.
   - **Learning Rate Scheduler**: Dynamically adjusts the learning rate based on validation loss to improve model performance.

### 3. **Training the Model**
   - Compiled using the `Adam` optimizer and `sparse_categorical_crossentropy` loss function.
   - Trained on an 80-20 train-validation split.
   - Early stopping and `ReduceLROnPlateau` callbacks are used to prevent overfitting and improve convergence.

### 4. **Evaluation and Results**
   - Evaluated on the test set with metrics like accuracy, precision, recall, and F1-score.
   - Confusion matrix and classification report are generated to understand class-level performance.
   - Plots for training/validation loss and accuracy over epochs are provided to monitor model performance.

## Getting Started

### Prerequisites
- Python 3.7+
- Keras and TensorFlow
- Numpy, Matplotlib, and Seaborn for visualization

Install the required packages using:
```bash
pip install tensorflow numpy matplotlib seaborn
