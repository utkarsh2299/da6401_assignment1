import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
        self.grad_w = None  # Gradient of weights
        self.grad_b = None  # Gradient of biases

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, grad):
        self.grad_w = np.dot(self.input.T, grad)
        self.grad_b = np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.weights.T)  # Gradient for previous layer

class Activation:
    def __init__(self, activation):
        self.activation = activation
    
    def forward(self, x):
        if self.activation == "relu":
            self.input = x
            return np.maximum(0, x)
        elif self.activation == "softmax":
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, grad):
        if self.activation == "relu":
            return grad * (self.input > 0)
        return grad  # Softmax handled in loss
