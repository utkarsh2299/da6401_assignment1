import numpy as np
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time

# Helper functions for activation functions and their derivatives
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))  # Clip to avoid overflow

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def softmax(z):
    # Subtract max for numerical stability
    exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# Loss function
def cross_entropy_loss(y_pred, y_true):
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # Calculate cross-entropy loss
    loss = -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
    return loss

# One-hot encoding helper function
def one_hot_encode(y, num_classes=10):
    encoded = np.zeros((y.shape[0], num_classes))
    for i, label in enumerate(y):
        encoded[i, label] = 1
    return encoded

# Neural Network class
class FeedforwardNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu'):
        """
        Initialize the neural network.
        
        Parameters:
        - input_size: Size of input features
        - hidden_layers: List of integers representing the size of each hidden layer
        - output_size: Number of output classes
        - activation: Activation function for hidden layers ('relu' or 'sigmoid')
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        
        # Choose activation function for hidden layers
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        
        # Initialize network parameters
        self.initialize_parameters()
        
        # Store activations and weighted inputs for backpropagation
        self.activations = []
        self.weighted_inputs = []
        
        # Store optimization parameters
        self.optimization_params = {}
        
    def initialize_parameters(self):
        """Initialize weights and biases for the network."""
        self.weights = []
        self.biases = []
        
        # Input layer to first hidden layer
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for weights
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale)
            # Initialize biases with zeros
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
        
    def forward_pass(self, X):
        """
        Perform forward pass through the network.
        
        Parameters:
        - X: Input data of shape (batch_size, input_size)
        
        Returns:
        - Output predictions after softmax
        """
        # Reset activation storage
        self.activations = [X]
        self.weighted_inputs = []
        
        # Hidden layers with chosen activation function
        current_activation = X
        for i in range(len(self.weights) - 1):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            self.weighted_inputs.append(z)
            current_activation = self.activation(z)
            self.activations.append(current_activation)
        
        # Output layer with softmax activation
        z_output = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        self.weighted_inputs.append(z_output)
        output_activation = softmax(z_output)
        self.activations.append(output_activation)
        
        return output_activation
    
    def backpropagation(self, y_true):
        """
        Compute gradients using backpropagation.
        
        Parameters:
        - y_true: True labels in one-hot encoded form
        
        Returns:
        - gradients for weights and biases
        """
        batch_size = y_true.shape[0]
        num_layers = len(self.weights)
        
        # Initialize gradients
        delta_weights = [np.zeros_like(w) for w in self.weights]
        delta_biases = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error (derivative of cross-entropy with softmax is simple)
        delta = self.activations[-1] - y_true
        
        # Backpropagate the error
        for l in reversed(range(num_layers)):
            delta_weights[l] = np.dot(self.activations[l].T, delta) / batch_size
            delta_biases[l] = np.sum(delta, axis=0, keepdims=True) / batch_size
            
            # Compute delta for the next layer (except for input layer)
            if l > 0:
                delta = np.dot(delta, self.weights[l].T) * self.activation_derivative(self.activations[l])
        
        return delta_weights, delta_biases
    
    def sgd(self, delta_weights, delta_biases, learning_rate=0.01):
        """Standard stochastic gradient descent update."""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * delta_weights[i]
            self.biases[i] -= learning_rate * delta_biases[i]
    
    def momentum(self, delta_weights, delta_biases, learning_rate=0.01, momentum=0.9):
        """Momentum-based gradient descent update."""
        if 'velocity_w' not in self.optimization_params:
            self.optimization_params['velocity_w'] = [np.zeros_like(w) for w in self.weights]
            self.optimization_params['velocity_b'] = [np.zeros_like(b) for b in self.biases]
        
        for i in range(len(self.weights)):
            self.optimization_params['velocity_w'][i] = momentum * self.optimization_params['velocity_w'][i] - learning_rate * delta_weights[i]
            self.optimization_params['velocity_b'][i] = momentum * self.optimization_params['velocity_b'][i] - learning_rate * delta_biases[i]
            
            self.weights[i] += self.optimization_params['velocity_w'][i]
            self.biases[i] += self.optimization_params['velocity_b'][i]
    
    def nesterov(self, delta_weights, delta_biases, learning_rate=0.01, momentum=0.9):
        """Nesterov accelerated gradient descent update."""
        if 'velocity_w' not in self.optimization_params:
            self.optimization_params['velocity_w'] = [np.zeros_like(w) for w in self.weights]
            self.optimization_params['velocity_b'] = [np.zeros_like(b) for b in self.biases]
        
        for i in range(len(self.weights)):
            old_velocity_w = self.optimization_params['velocity_w'][i].copy()
            old_velocity_b = self.optimization_params['velocity_b'][i].copy()
            
            self.optimization_params['velocity_w'][i] = momentum * self.optimization_params['velocity_w'][i] - learning_rate * delta_weights[i]
            self.optimization_params['velocity_b'][i] = momentum * self.optimization_params['velocity_b'][i] - learning_rate * delta_biases[i]
            
            # Apply Nesterov update
            self.weights[i] += -momentum * old_velocity_w + (1 + momentum) * self.optimization_params['velocity_w'][i]
            self.biases[i] += -momentum * old_velocity_b + (1 + momentum) * self.optimization_params['velocity_b'][i]
    
    def rmsprop(self, delta_weights, delta_biases, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        """RMSprop update rule."""
        if 'cache_w' not in self.optimization_params:
            self.optimization_params['cache_w'] = [np.zeros_like(w) for w in self.weights]
            self.optimization_params['cache_b'] = [np.zeros_like(b) for b in self.biases]
        
        for i in range(len(self.weights)):
            # Update cache with squared gradients
            self.optimization_params['cache_w'][i] = decay_rate * self.optimization_params['cache_w'][i] + (1 - decay_rate) * np.square(delta_weights[i])
            self.optimization_params['cache_b'][i] = decay_rate * self.optimization_params['cache_b'][i] + (1 - decay_rate) * np.square(delta_biases[i])
            
            # RMSprop update
            self.weights[i] -= learning_rate * delta_weights[i] / (np.sqrt(self.optimization_params['cache_w'][i]) + epsilon)
            self.biases[i] -= learning_rate * delta_biases[i] / (np.sqrt(self.optimization_params['cache_b'][i]) + epsilon)
    
    def adam(self, delta_weights, delta_biases, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Adam optimizer update rule."""
        if 't' not in self.optimization_params:
            self.optimization_params['t'] = 0
            self.optimization_params['m_w'] = [np.zeros_like(w) for w in self.weights]
            self.optimization_params['m_b'] = [np.zeros_like(b) for b in self.biases]
            self.optimization_params['v_w'] = [np.zeros_like(w) for w in self.weights]
            self.optimization_params['v_b'] = [np.zeros_like(b) for b in self.biases]
        
        self.optimization_params['t'] += 1
        t = self.optimization_params['t']
        
        for i in range(len(self.weights)):
            # Update biased first moment estimate
            self.optimization_params['m_w'][i] = beta1 * self.optimization_params['m_w'][i] + (1 - beta1) * delta_weights[i]
            self.optimization_params['m_b'][i] = beta1 * self.optimization_params['m_b'][i] + (1 - beta1) * delta_biases[i]
            
            # Update biased second raw moment estimate
            self.optimization_params['v_w'][i] = beta2 * self.optimization_params['v_w'][i] + (1 - beta2) * np.square(delta_weights[i])
            self.optimization_params['v_b'][i] = beta2 * self.optimization_params['v_b'][i] + (1 - beta2) * np.square(delta_biases[i])
            
            # Compute bias-corrected first moment estimate
            m_w_corrected = self.optimization_params['m_w'][i] / (1 - beta1**t)
            m_b_corrected = self.optimization_params['m_b'][i] / (1 - beta1**t)
            
            # Compute bias-corrected second raw moment estimate
            v_w_corrected = self.optimization_params['v_w'][i] / (1 - beta2**t)
            v_b_corrected = self.optimization_params['v_b'][i] / (1 - beta2**t)
            
            # Adam update
            self.weights[i] -= learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + epsilon)
            self.biases[i] -= learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + epsilon)
    
    def nadam(self, delta_weights, delta_biases, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Nadam (Nesterov-accelerated Adam) optimizer update rule."""
        if 't' not in self.optimization_params:
            self.optimization_params['t'] = 0
            self.optimization_params['m_w'] = [np.zeros_like(w) for w in self.weights]
            self.optimization_params['m_b'] = [np.zeros_like(b) for b in self.biases]
            self.optimization_params['v_w'] = [np.zeros_like(w) for w in self.weights]
            self.optimization_params['v_b'] = [np.zeros_like(b) for b in self.biases]
        
        self.optimization_params['t'] += 1
        t = self.optimization_params['t']
        
        for i in range(len(self.weights)):
            # Update biased first moment estimate
            self.optimization_params['m_w'][i] = beta1 * self.optimization_params['m_w'][i] + (1 - beta1) * delta_weights[i]
            self.optimization_params['m_b'][i] = beta1 * self.optimization_params['m_b'][i] + (1 - beta1) * delta_biases[i]
            
            # Update biased second raw moment estimate
            self.optimization_params['v_w'][i] = beta2 * self.optimization_params['v_w'][i] + (1 - beta2) * np.square(delta_weights[i])
            self.optimization_params['v_b'][i] = beta2 * self.optimization_params['v_b'][i] + (1 - beta2) * np.square(delta_biases[i])
            
            # Compute bias-corrected first moment estimate
            m_w_corrected = self.optimization_params['m_w'][i] / (1 - beta1**t)
            m_b_corrected = self.optimization_params['m_b'][i] / (1 - beta1**t)
            
            # Compute bias-corrected second raw moment estimate
            v_w_corrected = self.optimization_params['v_w'][i] / (1 - beta2**t)
            v_b_corrected = self.optimization_params['v_b'][i] / (1 - beta2**t)
            
            # Calculate the Nesterov momentum term
            m_w_nesterov = beta1 * m_w_corrected + (1 - beta1) * delta_weights[i] / (1 - beta1**t)
            m_b_nesterov = beta1 * m_b_corrected + (1 - beta1) * delta_biases[i] / (1 - beta1**t)
            
            # Nadam update
            self.weights[i] -= learning_rate * m_w_nesterov / (np.sqrt(v_w_corrected) + epsilon)
            self.biases[i] -= learning_rate * m_b_nesterov / (np.sqrt(v_b_corrected) + epsilon)
    
    def update_parameters(self, delta_weights, delta_biases, optimizer='sgd', **kwargs):
        """
        Update network parameters based on the chosen optimizer.
        
        Parameters:
        - delta_weights: Weight gradients
        - delta_biases: Bias gradients
        - optimizer: Optimization algorithm ('sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam')
        - **kwargs: Additional parameters for the optimizer
        """
        if optimizer == 'sgd':
            self.sgd(delta_weights, delta_biases, **kwargs)
        elif optimizer == 'momentum':
            self.momentum(delta_weights, delta_biases, **kwargs)
        elif optimizer == 'nesterov':
            self.nesterov(delta_weights, delta_biases, **kwargs)
        elif optimizer == 'rmsprop':
            self.rmsprop(delta_weights, delta_biases, **kwargs)
        elif optimizer == 'adam':
            self.adam(delta_weights, delta_biases, **kwargs)
        elif optimizer == 'nadam':
            self.nadam(delta_weights, delta_biases, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, 
              optimizer='sgd', learning_rate=0.01, verbose=True, **kwargs):
        """
        Train the neural network.
        
        Parameters:
        - X_train: Training data
        - y_train: Training labels (one-hot encoded)
        - X_val: Validation data
        - y_val: Validation labels (one-hot encoded)
        - epochs: Number of training epochs
        - batch_size: Mini-batch size
        - optimizer: Optimization algorithm
        - learning_rate: Learning rate
        - verbose: Whether to print progress
        - **kwargs: Additional parameters for the optimizer
        
        Returns:
        - history: Dictionary containing loss and accuracy history
        """
        # Convert labels to one-hot encoding if needed
        if len(y_train.shape) == 1:
            y_train = one_hot_encode(y_train, self.output_size)
        
        if X_val is not None and y_val is not None and len(y_val.shape) == 1:
            y_val = one_hot_encode(y_val, self.output_size)
        
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # Initialize history dictionary
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # For each epoch
        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0
            
            # Shuffle the training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward_pass(X_batch)
                
                # Calculate loss
                batch_loss = cross_entropy_loss(y_pred, y_batch)
                epoch_loss += batch_loss * (end_idx - start_idx) / n_samples
                
                # Backward pass
                delta_weights, delta_biases = self.backpropagation(y_batch)
                
                # Update parameters
                self.update_parameters(delta_weights, delta_biases, optimizer, learning_rate=learning_rate, **kwargs)
            
            # Calculate training accuracy
            y_pred_train = self.predict(X_train)
            train_acc = accuracy_score(np.argmax(y_train, axis=1), y_pred_train)
            
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(train_acc)
            
            # Calculate validation metrics if validation data is provided
            if X_val is not None and y_val is not None:
                y_pred_val = self.predict(X_val)
                val_acc = accuracy_score(np.argmax(y_val, axis=1), y_pred_val)
                val_loss = cross_entropy_loss(one_hot_encode(y_pred_val, self.output_size), y_val)
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - {time.time()-start_time:.2f}s - loss: {epoch_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - {time.time()-start_time:.2f}s - loss: {epoch_loss:.4f} - acc: {train_acc:.4f}")
        
        return history
    
    def predict(self, X):
        """
        Make predictions for input data.
        
        Parameters:
        - X: Input data
        
        Returns:
        - Predicted class labels
        """
        # Forward pass
        y_pred = self.forward_pass(X)
        
        # Return class with highest probability
        return np.argmax(y_pred, axis=1)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data.
        
        Parameters:
        - X: Test data
        - y: Test labels
        
        Returns:
        - accuracy: Classification accuracy
        """
        y_pred = self.predict(X)
        
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)
            
        return accuracy_score(y, y_pred)

# Data preprocessing
def preprocess_data():
    """Load and preprocess the Fashion-MNIST dataset."""
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

# Example usage
def main():
    # Fashion-MNIST class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data()
    
    # Print dataset shapes
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    # Define model architecture
    input_size = 784  # 28x28 pixels
    hidden_layers = [128, 64]  # Two hidden layers
    output_size = 10  # 10 classes in Fashion-MNIST
    
    # Create and train the model
    model = FeedforwardNeuralNetwork(input_size, hidden_layers, output_size, activation='relu')
    
    # Train with SGD
    print("\nTraining with SGD:")
    history_sgd = model.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=64, 
                          optimizer='sgd', learning_rate=0.01, verbose=True)
    
    # Evaluate the model
    test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy with SGD: {test_acc:.4f}")
    
    # Create a new model for a different optimizer
    model = FeedforwardNeuralNetwork(input_size, hidden_layers, output_size, activation='relu')
    
    # Train with Adam
    print("\nTraining with Adam:")
    history_adam = model.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=64, 
                           optimizer='adam', learning_rate=0.001, verbose=True)
    
    # Evaluate the model
    test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy with Adam: {test_acc:.4f}")
    
    # Visualize training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_sgd['train_acc'], label='SGD Train')
    plt.plot(history_sgd['val_acc'], label='SGD Validation')
    plt.plot(history_adam['train_acc'], label='Adam Train')
    plt.plot(history_adam['val_acc'], label='Adam Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_sgd['train_loss'], label='SGD Train')
    plt.plot(history_sgd['val_loss'], label='SGD Validation')
    plt.plot(history_adam['train_loss'], label='Adam Train')
    plt.plot(history_adam['val_loss'], label='Adam Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Make some predictions
    sample_indices = np.random.choice(len(X_test), 5)
    sample_images = X_test[sample_indices]
    sample_labels = y_test[sample_indices]
    
    predictions = model.predict(sample_images)
    
    # Display sample images and predictions
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(sample_indices):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[predictions[i]]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()