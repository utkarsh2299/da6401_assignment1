import numpy as np
import time
import wandb
import pickle
from utils.activations import get_activation_function, get_activation_derivative
from utils.losses import cross_entropy_loss
from optimizers.optimizers import get_optimizer


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
        # Store network architecture parameters
        self.input_size = input_size  # Number of input features
        self.hidden_layers = hidden_layers  # List containing size of each hidden layer
        self.output_size = output_size  # Number of output classes/neurons
        
        # Get activation function and its derivative from utility functions
        # These will be used during forward and backward passes
        self.activation = get_activation_function(activation)  # Function to apply during forward pass
        self.activation_derivative = get_activation_derivative(activation)  # Used during backpropagation
        
        # Initialize network parameters (weights and biases)
        self.initialize_parameters()
        
        # Storage for intermediate values during forward pass
        # These are needed for backpropagation
        self.activations = []  # Store output values after activation functions
        self.weighted_inputs = []  # Store values before activation functions (z = wx + b)
        
        # Dictionary to store optimizer-specific parameters (momentum, rmsprop cache, etc.)
        self.optimization_params = {}
        
    def initialize_parameters(self):
        """
        Initialize weights and biases for the network using He initialization.
        He initialization helps with training deeper networks, especially with ReLU activations.
        """
        self.weights = []  # List to store weight matrices
        self.biases = []  # List to store bias vectors
        
        # Create a list of all layer sizes (input, hidden, output)
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        
        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            # He initialization scale factor: sqrt(2/n_inputs)
            # This prevents vanishing/exploding gradients in deep networks
            scale = np.sqrt(2.0 / layer_sizes[i])
            
            # Create weight matrix with random values scaled by He factor
            # Shape: (current_layer_size, next_layer_size)
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale)
            
            # Initialize biases with zeros
            # Shape: (1, next_layer_size) - broadcasting will apply to all samples
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
        
    def forward_pass(self, X):
        """
        Perform forward pass through the network.
        
        Parameters:
        - X: Input data of shape (batch_size, input_size)
        
        Returns:
        - Output predictions after softmax
        """
        # Reset storage for activations and weighted inputs
        # Important for each new forward pass
        self.activations = [X]  # Input is the first "activation"
        self.weighted_inputs = []  # Store z values (wx + b) for backprop
        
        # Process through hidden layers with chosen activation function
        current_activation = X  # Start with input data
        
        # Process all layers except the output layer
        for i in range(len(self.weights) - 1):
            # Compute weighted input: z = wx + b
            a = np.dot(current_activation, self.weights[i]) + self.biases[i]
            self.weighted_inputs.append(a)  # Store for backpropagation
            
            # Apply activation function
            current_activation = self.activation(a)
            self.activations.append(current_activation)  # Store for backpropagation
        
        # Special handling for output layer (using softmax)
        z_output = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        self.weighted_inputs.append(z_output)
        
        # Apply softmax activation for output layer
        # Subtracting max value for numerical stability (prevents overflow)
        exp_scores = np.exp(z_output - np.max(z_output, axis=1, keepdims=True))
        # Normalize to get probabilities
        output_activation = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Store final output activations
        self.activations.append(output_activation)
        
        return output_activation
    
    def backpropagation(self, y_true):
        """
        Compute gradients using backpropagation algorithm.
        
        Parameters:
        - y_true: True labels in one-hot encoded form
        
        Returns:
        - gradients for weights and biases (delta_weights, delta_biases)
        """
        batch_size = y_true.shape[0]  # Number of samples in current batch
        num_layers = len(self.weights)  # Total number of weight matrices (layers)
        
        # Initialize gradient lists with zeros (same shape as weights and biases)
        delta_weights = [np.zeros_like(w) for w in self.weights]
        delta_biases = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error calculation
        # For softmax + cross-entropy, the derivative simplifies to (output - target)
        # This is a nice mathematical property that makes the code simpler
        delta = self.activations[-1] - y_true  # Output error
        
        # Backpropagate the error through the network (right to left)
        for l in reversed(range(num_layers)):
            # Calculate weight gradients: (layer_activation.T @ delta) / batch_size
            # Matrix multiplication of activations and error
            delta_weights[l] = np.dot(self.activations[l].T, delta) / batch_size
            
            # Calculate bias gradients: mean of error across batch
            delta_biases[l] = np.sum(delta, axis=0, keepdims=True) / batch_size
            
            # Compute delta for previous layer (except for input layer)
            if l > 0:
                # delta = (delta @ weights.T) * activation_derivative(activation)
                # Element-wise multiplication with activation derivative
                delta = np.dot(delta, self.weights[l].T) * self.activation_derivative(self.activations[l])
        
        return delta_weights, delta_biases
    
    def update_parameters(self, delta_weights, delta_biases, optimizer='sgd', **kwargs):
        """
        Update network parameters based on the chosen optimizer.
        
        Parameters:
        - delta_weights: Weight gradients from backpropagation
        - delta_biases: Bias gradients from backpropagation
        - optimizer: Optimization algorithm ('sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam')
        - **kwargs: Additional parameters for the optimizer (learning_rate, beta1, beta2, etc.)
        """
        # Get the appropriate optimizer function
        optimizer_func = get_optimizer(optimizer)
        
        # Apply the optimizer to update weights and biases
        # Pass the current weights, biases, gradients, and optimization state
        optimizer_func(
            self.weights, self.biases,  # Parameters to update
            delta_weights, delta_biases,  # Computed gradients
            self.optimization_params,  # Optimizer state (momentum, etc.)
            **kwargs  # Additional parameters like learning rate
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, 
              optimizer='sgd', learning_rate=0.01, verbose=True, use_wandb=False, **kwargs):
        """
        Train the neural network using mini-batch gradient descent.
        
        Parameters:
        - X_train: Training data features
        - y_train: Training labels (one-hot encoded or class indices)
        - X_val: Validation data features (optional)
        - y_val: Validation labels (optional)
        - epochs: Number of training epochs
        - batch_size: Mini-batch size for stochastic optimization
        - optimizer: Optimization algorithm to use
        - learning_rate: Learning rate for optimization
        - verbose: Whether to print progress after each epoch
        - use_wandb: Whether to log metrics to Weights & Biases
        - **kwargs: Additional parameters for the optimizer
        
        Returns:
        - history: Dictionary containing loss and accuracy history
        """
        # Convert labels to one-hot encoding if they're not already
        if len(y_train.shape) == 1:
            y_train = self._one_hot_encode(y_train)
        
        # Also convert validation labels if provided
        if X_val is not None and y_val is not None and len(y_val.shape) == 1:
            y_val = self._one_hot_encode(y_val)
        
        # Calculate number of samples and batches
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))  # Ceiling division for partial batches
        
        # Initialize history dictionary to track metrics
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Training loop - for each epoch
        for epoch in range(epochs):
            start_time = time.time()  # Track time per epoch
            epoch_loss = 0  # Reset loss for this epoch
            
            # Shuffle the training data for stochastic optimization
            # This helps prevent getting stuck in local minima
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training loop
            for batch in range(n_batches):
                # Get batch indices
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, n_samples)  # Handle last batch being smaller
                
                # Extract current batch
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass - get predictions
                y_pred = self.forward_pass(X_batch)
                
                # Calculate loss for this batch
                batch_loss = cross_entropy_loss(y_pred, y_batch)
                # Update epoch loss (weighted by batch size)
                epoch_loss += batch_loss * (end_idx - start_idx) / n_samples
                
                # Backward pass - compute gradients
                delta_weights, delta_biases = self.backpropagation(y_batch)
                
                # Update network parameters using chosen optimizer
                self.update_parameters(
                    delta_weights, delta_biases,  # Gradients
                    optimizer=optimizer,  # Optimization algorithm
                    learning_rate=learning_rate,  # Learning rate
                    **kwargs  # Additional optimizer parameters
                )
            
            # Calculate training accuracy after epoch
            y_pred_train = self.predict(X_train)  # Get class predictions
            y_true_train = np.argmax(y_train, axis=1)  # Convert one-hot to class indices
            train_acc = np.mean(y_pred_train == y_true_train)  # Calculate accuracy
            
            # Store training metrics
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(train_acc)
            
            # Calculate validation metrics if validation data is provided
            if X_val is not None and y_val is not None:
                # Get validation predictions and convert to class indices
                y_pred_val = self.predict(X_val)
                y_true_val = np.argmax(y_val, axis=1)
                val_acc = np.mean(y_pred_val == y_true_val)  # Calculate accuracy
                
                # For loss calculation, we need one-hot encoded predictions
                y_pred_val_one_hot = self._one_hot_encode(y_pred_val)
                val_loss = cross_entropy_loss(y_pred_val_one_hot, y_val)
                
                # Store validation metrics
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Print progress if verbose mode is enabled
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - {time.time()-start_time:.2f}s - loss: {epoch_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
                
                # Log metrics to Weights & Biases if enabled
                if use_wandb:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": epoch_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc
                    })
            else:
                # Print progress without validation metrics
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - {time.time()-start_time:.2f}s - loss: {epoch_loss:.4f} - acc: {train_acc:.4f}")
                
                # Log training metrics to Weights & Biases if enabled
                if use_wandb:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": epoch_loss,
                        "train_acc": train_acc
                    })
        
        # Return the history dictionary for analysis
        return history
    
    def predict(self, X):
        """
        Make predictions for input data.
        
        Parameters:
        - X: Input data features
        
        Returns:
        - Predicted class labels (indices, not one-hot)
        """
        # Forward pass to get class probabilities
        y_pred = self.forward_pass(X)
        
        # Return class index with highest probability for each sample
        return np.argmax(y_pred, axis=1)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data.
        
        Parameters:
        - X: Test data features
        - y: Test labels (can be one-hot or indices)
        
        Returns:
        - accuracy: Classification accuracy
        """
        # Get model predictions
        y_pred = self.predict(X)
        
        # Convert y to class indices if it's one-hot encoded
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)
            
        # Return mean accuracy (fraction of correct predictions)
        return np.mean(y_pred == y)
    
    def _one_hot_encode(self, y, num_classes=None):
        """
        One-hot encode the target labels.
        
        Parameters:
        - y: Target labels as indices
        - num_classes: Number of classes (defaults to output_size)
        
        Returns:
        - One-hot encoded labels matrix
        """
        # If num_classes not specified, use model's output size
        if num_classes is None:
            num_classes = self.output_size
        
        # Create zero matrix of shape (n_samples, n_classes)
        encoded = np.zeros((y.shape[0], num_classes))
        
        # Set the appropriate index to 1 for each sample
        for i, label in enumerate(y):
            encoded[i, label] = 1
            
        return encoded
    
    def save(self, filename):
        """
        Save the model to a file using pickle.
        
        Parameters:
        - filename: Name of the file to save the model
        """
        # Open file in binary write mode
        with open(filename, 'wb') as f:
            # Serialize and save the entire model
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename):
        """
        Load the model from a file.
        
        Parameters:
        - filename: Name of the file to load the model from
        
        Returns:
        - Loaded model instance
        """
        # Open file in binary read mode
        with open(filename, 'rb') as f:
            # Deserialize and return the model
            return pickle.load(f)