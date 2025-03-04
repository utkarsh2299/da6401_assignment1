import numpy as np
from keras.datasets import fashion_mnist
# from tensorflow.keras.datasets import fashion_mnist
import numpy as np

def preprocess_data():
    """
    Load and preprocess the Fashion-MNIST dataset with 10% of the training data set aside for validation.
    
    Returns:
    - X_train: Training data
    - y_train: Training labels
    - X_val: Validation data
    - y_val: Validation labels
    - X_test: Test data
    - y_test: Test labels
    """
    # Load Fashion-MNIST data
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Reshape and normalize data
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    
    # validation size (10% of training data)
    val_size = int(0.1 * X_train.shape[0])  # 10% of the training data
    
    # Shuffle data before splitting
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)

    X_train, y_train = X_train[indices], y_train[indices]
    
    # Split training set to create validation set
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_train, y_train = X_train[val_size:], y_train[val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def one_hot_encode(y, num_classes=10):
    """
    One-hot encode the target labels.
    
    Parameters:
    - y: Target labels
    - num_classes: Number of classes
    
    Returns:
    - encoded: One-hot encoded labels
    """
    encoded = np.zeros((y.shape[0], num_classes))
    for i, label in enumerate(y):
        encoded[i, label] = 1
    return encoded

