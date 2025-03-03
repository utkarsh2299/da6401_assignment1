import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function.
    
    Parameters:
    - z: Input tensor
    
    Returns:
    - Sigmoid activation: 1 / (1 + exp(-z))
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))  # Clip to avoid overflow

def sigmoid_derivative(a):
    """
    Derivative of the sigmoid function.
    
    Parameters:
    - a: Sigmoid activation
    
    Returns:
    - Derivative: a * (1 - a)
    """
    return a * (1 - a)

def relu(z):
    """
    ReLU activation function.
    
    Parameters:
    - z: Input tensor
    
    Returns:
    - ReLU activation: max(0, z)
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """
    Derivative of the ReLU function.
    
    Parameters:
    - z: Input tensor
    
    Returns:
    - Derivative: 1 if z > 0, 0 otherwise
    """
    return np.where(z > 0, 1, 0)

def tanh(z):
    """
    Tanh activation function.
    
    Parameters:
    - z: Input tensor
    
    Returns:
    - Tanh activation: (e^z - e^(-z)) / (e^z + e^(-z))
    """
    return np.tanh(z)

def tanh_derivative(a):
    """
    Derivative of the tanh function.
    
    Parameters:
    - a: Tanh activation
    
    Returns:
    - Derivative: 1 - a^2
    """
    # return 1 - np.power(a, 2)
    return 1 - np.tanh(a)**2


def softmax(z):
    """
    Softmax activation function.
    
    Parameters:
    - z: Input tensor
    
    Returns:
    - Softmax activation: exp(z) / sum(exp(z))
    """
    # Subtract max for numerical stability
    exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def get_activation_function(name):
    """
    Get activation function by name.
    
    Parameters:
    - name: Name of the activation function
    
    Returns:
    - activation_function: The activation function
    """
    if name.lower() == 'sigmoid':
        return sigmoid
    elif name.lower() == 'relu':
        return relu
    elif name.lower() == 'tanh':
        return tanh
    elif name.lower() == 'softmax':
        return softmax
    else:
        raise ValueError(f"Unknown activation function: {name}")

def get_activation_derivative(name):
    """
    Get derivative of activation function by name.
    
    Parameters:
    - name: Name of the activation function
    
    Returns:
    - activation_derivative: The derivative of the activation function
    """
    if name.lower() == 'sigmoid':
        return sigmoid_derivative
    elif name.lower() == 'relu':
        return relu_derivative
    elif name.lower() == 'tanh':
        return tanh_derivative
    else:
        raise ValueError(f"Unknown activation function derivative: {name}")
