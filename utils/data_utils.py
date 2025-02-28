import numpy as np
from keras.datasets import fashion_mnist

def preprocess_data():
    """
    Load and preprocess the Fashion-MNIST dataset.
    
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
    
    # Split training set to create validation set
    val_size = 10000
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
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

def data_generator(X, y, batch_size):
    """
    Generate batches of data for training.
    
    Parameters:
    - X: Input data
    - y: Target labels
    - batch_size: Size of each batch
    
    Yields:
    - X_batch: Batch of input data
    - y_batch: Batch of target labels
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield X[batch_indices], y[batch_indices]