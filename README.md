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
   - Normalize the images by scaling pixel values to be between 0 and 1, which helps in stabilizing and accelerating the training process.
   - Flatten the 28x28 images into 784-dimensional vectors since we're using an ANN that accepts 1D input.

### 2. **Model Architecture**
   - We use a sequential neural network with three fully connected (dense) layers:
     - **Input Layer**: Accepts a 784-dimensional flattened vector.
     - **Hidden Layers**: Three dense layers with 512, 256, 64, and 32 neurons respectively, and batch normalization and dropout layer to prevent overfitting.
     - **Output Layer**: 10 neurons with softmax activation to output probabilities for each class.
   
   - **Regularization Techniques**:
     - **Dropout Layers**: Dropout is applied to each hidden layer with a dropout rate of 30%. This randomly disables 30% of neurons during each update, helping prevent overfitting by ensuring the network does not rely too heavily on any single neuron.
     - **Batch Normalization**: Applied after each hidden layer to normalize the output of the previous layer. Batch normalization improves training speed and stability by maintaining a consistent mean and variance of activations.
     - **Learning Rate Scheduler**: `ReduceLROnPlateau` callback is used to lower the learning rate if the validation loss plateaus, which helps in achieving more accurate training without drastic fluctuations.
   
   - **Learning Rate Schedule and Early Stopping**:
     - The learning rate is reduced automatically if there’s no improvement in validation loss for a specified number of epochs (patience=3).
     - **Early Stopping**: Stops training if the validation loss does not improve for a given number of epochs (patience=5) and restores the model to its best weights, helping avoid overfitting.

### 3. **Training the Model**
   - **Compilation**: The model is compiled with the `Adam` optimizer and the `sparse_categorical_crossentropy` loss function (suitable for multiclass classification with integer labels).
   - **Training Setup**: 
     - Trained on an 80-20 split of the training data (60,000 images) for training and validation.
     - **Batch Size**: 64, to balance between memory efficiency and training speed.
     - **Callbacks**: Early stopping and ReduceLROnPlateau are applied to improve model performance and avoid overfitting.
   - **Epochs**: The model is trained for up to 100 epochs, though early stopping may stop training sooner.

### 4. **Evaluation and Results**
   - **Evaluation on Test Set**: The model is evaluated on the 10,000-image test set, with metrics like accuracy, precision, recall, and F1-score calculated for each class.
   - **Confusion Matrix**: Generated to provide a clear picture of where the model is misclassifying across different classes.
   - **Loss and Accuracy Curves**: Training and validation loss and accuracy are plotted over epochs to monitor overfitting and convergence.

## Getting Started

### Prerequisites
- Python 3.7+
- Keras and TensorFlow
- Numpy, Matplotlib, and Seaborn for visualization

Install the required packages using:
```bash
pip install tensorflow numpy matplotlib seaborn
