import numpy as np

def one_hot_encode(y: np.array, num_classes = 10):
    '''It converts class labels (integers) into one-hot encoded vectors,
        which are necessary for training a neural network in a classification task.'''
    # num_classes = 10
    ysize=y.shape[0]
    one_hot_vector = np.zeros((ysize, num_classes))  #Creates a matrix of zeros (y_samples × num_classes)
    one_hot_vector[np.arange(ysize), y] = 1 # Sets the correct class index to 1 for each sample
    return one_hot_vector

def preprocess_data(X_train, X_test):
    '''
    Since Fashion-MNIST images are 28x28 grayscale, this reshapes them into flat 1D vectors of size 784 (28 × 28).
    This is needed as nn expects 1D feature vectors rather than 2D images.
    Dividing by 255.0 scales pixel values from [0, 255] to [0, 1]. This helps the neural network learn more efficiently 
    by keeping input values small, preventing large weight updates.
    '''
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    return X_train, X_test

def compute_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

def compute_accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
