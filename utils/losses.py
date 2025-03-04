import numpy as np

def cross_entropy_loss(y_pred, y_true, X_batch):
    """
    Calculate cross-entropy loss.
    
    Parameters:
    - y_pred: Predicted probabilities
    - y_true: True labels (one-hot encoded)
    
    Returns:
    - Cross-entropy loss value
    """
    # Add small epsilon to avoid log(0)
    # epsilon = 1e-15
    # y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # Calculate cross-entropy loss
    loss = -np.sum(y_true * np.log(y_pred + 1e-10)) / X_batch.shape[0]
    return loss

def mean_squared_error(y_pred, y_true):
    """
    Calculate mean squared error loss.
    
    Parameters:
    - y_pred: Predicted values
    - y_true: True values
    
    Returns:
    - MSE loss value
    """
    return np.mean(np.square(y_pred - y_true))

def binary_cross_entropy(y_pred, y_true):
    """
    Calculate binary cross-entropy loss.
    
    Parameters:
    - y_pred: Predicted probabilities
    - y_true: True labels (0 or 1)
    
    Returns:
    - Binary cross-entropy loss value
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Dictionary mapping loss function names to functions
LOSS_FUNCTIONS = {
    'cross_entropy': cross_entropy_loss,
    'mse': mean_squared_error,
    'binary_cross_entropy': binary_cross_entropy
}

def get_loss_function(name):
    """
    Get loss function by name.
    
    Parameters:
    - name: Name of loss function
    
    Returns:
    - Loss function
    """
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_FUNCTIONS[name]