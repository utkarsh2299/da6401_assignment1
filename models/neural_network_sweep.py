import numpy as np
import time
import wandb
from utils.activations import get_activation_function, get_activation_derivative
from utils.losses import get_loss_function

class FeedforwardNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', weight_init='xavier', weight_decay=0, loss="cross_entropy_loss", learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=0.5):
        """
        Initialize a feedforward neural network.
        
        Parameters:
        - input_size: Number of input features
        - hidden_layers: List of hidden layer sizes
        - output_size: Number of output classes
        - activation: Activation function ('relu', 'sigmoid', or 'tanh')
        - weight_init: Weight initialization method ('random' or 'xavier')
        - weight_decay: L2 regularization parameter
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        # self.activation = activation
        self.weight_decay = weight_decay
        self.activation_fn = get_activation_function(activation)  # Function to apply during forward pass
        self.activation_derivative = get_activation_derivative(activation)  # Used during backpropagation
        self.softmax=get_activation_function("softmax")
        # Initialize weights and biases
        self.weights = []  # List to store weight matrices
        self.biases = []   # List to store bias vectors       
        
        # self.initialize_parameters(weight_init)
        # Input layer to first hidden layer
        if weight_init == 'xavier':
            self.weights.append(np.random.randn(input_size, hidden_layers[0]) * np.sqrt(1.0 / input_size))
        elif weight_init == 'he':
            self.weights.append(np.random.randn(input_size, hidden_layers[0]) * np.sqrt(2.0 / input_size))
        else:  # random
            self.weights.append(np.random.randn(input_size, hidden_layers[0]) * 0.01)
        self.biases.append(np.zeros((1, hidden_layers[0])))

        # Hidden layers
        for i in range(1, len(hidden_layers)):
            if weight_init == 'xavier':
                self.weights.append(np.random.randn(hidden_layers[i-1], hidden_layers[i]) * np.sqrt(1.0 / hidden_layers[i-1]))
            elif weight_init == 'he':
                self.weights.append(np.random.randn(hidden_layers[i-1], hidden_layers[i]) * np.sqrt(2.0 / hidden_layers[i-1]))
            else:  # random
                self.weights.append(np.random.randn(hidden_layers[i-1], hidden_layers[i]) * 0.01)
            self.biases.append(np.zeros((1, hidden_layers[i])))

        # Last hidden layer to output layer
        if weight_init == 'xavier':
            self.weights.append(np.random.randn(hidden_layers[-1], output_size) * np.sqrt(1.0 / hidden_layers[-1]))
        elif weight_init == 'he':
            self.weights.append(np.random.randn(hidden_layers[-1], output_size) * np.sqrt(2.0 / hidden_layers[-1]))
        else:  # random
            self.weights.append(np.random.randn(hidden_layers[-1], output_size) * 0.01)
        self.biases.append(np.zeros((1, output_size)))
                
        
        
        self.loss_function = get_loss_function(loss)
        # if self.loss_function=="cross_entropy_loss":
        #     self.loss_function = get_loss_function("cross_entropy_loss")
        # elif self.loss_function == "mean_squared_error":
        #     self.loss_function = get_loss_function("mean_squared_error")
        self.optimization_params={}
        self.learning_rate = learning_rate
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon = epsilon
        self.decay_rate= decay_rate

       
    def initialize_parameters(self, weight_init='he'):
        """
        Initialize weights and biases for the network using He, Xavier, or Random initialization.

        Parameters:
        - weight_init (str): The initialization method. Options aree:
            'he'      -> He initialization (recommended for ReLU activations)
            'xavier'  -> Xavier initialization (recommended for Tanh activations)
            'random'  -> Small random values (not recommended for deep networks)
        """
        self.weights = []  # List to store weight matrices
        self.biases = []   # List to store bias vectors
        
        # Create a list of all layer sizes (input, hidden, output)
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        
        # Initialize weights and biases for each layer
        # Use Xavier for Sigmoid/Tanh networks (balanced activations). Add to report
        # Use He for ReLU-based networks (avoids shrinking activations).
        for i in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i + 1]
            
            if weight_init == 'he':
                scale = np.sqrt(2.0 / input_dim)  # He initialization
            elif weight_init == 'xavier':
                scale = np.sqrt(1.0 / input_dim)  # Xavier initialization
            else:  # 'random'
                scale = 0.01  # Small random values
                
            # Create weight matrix with chosen initialization
            self.weights.append(np.random.randn(input_dim, output_dim) * scale)
            
            # Initialize biases with zeros
            self.biases.append(np.zeros((1, output_dim)))
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Parameters:
        - X: Input data
        
        Returns:
        - activations: List of activations for each layer
        - layer_inputs: List of inputs to each layer (pre-activation)
        """
        activations = [X]
        layer_inputs = []
        
        # Input layer to first hidden layer
        z = np.dot(X, self.weights[0]) + self.biases[0]
        layer_inputs.append(z)
        a = self.activation_fn(z)
        activations.append(a)
        
        # Hiddens layers
        for i in range(1, len(self.hidden_layers)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            layer_inputs.append(z)
            a = self.activation_fn(z)
            activations.append(a)
        
        # Output layer
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        layer_inputs.append(z)
        
        a = self.softmax(z)
        activations.append(a)
        
        return activations, layer_inputs
    
    def backward(self, X, y, activations, layer_inputs, optimizer, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        """
        Backward pass (backpropagation).
        
        Parameters:
        - X: Input data
        - y: Target labels (one-hot encoded)
        - activations: List of activations from forward pass
        - layer_inputs: List of layer inputs from forward pass
        - optimizer: Optimization algorithm
        - learning_rate: Learning rate
        - beta1, beta2, epsilon: Optimizer parameters for Adam/RMSProp
        
        Returns:
        - weight_gradients: Gradients of weights
        - bias_gradients: Gradients of biases
        """
        m = X.shape[0]
        num_layers = len(self.weights)
        
        # Output layer error
        delta = activations[-1] - y
        
        # Initialize gradients
        weight_gradients = [None] * num_layers
        bias_gradients = [None] * num_layers
        
        # Output layer gradients
        weight_gradients[-1] = np.dot(activations[-2].T, delta) / m
        bias_gradients[-1] = np.sum(delta, axis=0, keepdims=True) / m
        
        # Add L2 regularization
        if self.weight_decay > 0:
            weight_gradients[-1] += self.weight_decay * self.weights[-1]
        
        # Backpropagate the error
        for l in range(num_layers - 2, -1, -1):
            delta = np.dot(delta, self.weights[l+1].T) * self.activation_derivative(layer_inputs[l])
            weight_gradients[l] = np.dot(activations[l].T, delta) / m
            bias_gradients[l] = np.sum(delta, axis=0, keepdims=True) / m
            
            # Add L2 regularization
            if self.weight_decay > 0:
                weight_gradients[l] += self.weight_decay * self.weights[l]
        
   
        clip_value=5.0  #gradient clip to prevent exploding gradients and overflow
        for l in range(num_layers):
            weight_gradients[l] = np.clip(weight_gradients[l], -clip_value, clip_value)
            bias_gradients[l] = np.clip(bias_gradients[l], -clip_value, clip_value)
        
        from optimizers.optimizers import get_optimizer
        
        optimizer_func = get_optimizer(optimizer)
        optimizer_func(
            self.weights, self.biases,  # Parameters to update
            weight_gradients, bias_gradients,  # Computed gradients
            self.optimization_params,  # Optimizer state (momentum, etc.)
            **kwargs  
        )    
        
        return weight_gradients, bias_gradients
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, optimizer='adam', learning_rate=0.001, verbose=True, use_wandb=False):
        """
        Train the neural network.
        
        Parameters:
        - X_train: Training data
        - y_train: Training labels
        - X_val: Validation data
        - y_val: Validation labels
        - epochs: Number of epochs
        - batch_size: Batch size
        - optimizer: Optimization algorithm
        - learning_rate: Learning rate
        - verbose: Whether to print progress
        - use_wandb: Whether to log to wandb
        
        Returns:
        - history: Training history
        """
        if use_wandb == True:
            wandb.init()
        # One-hot encode labels
        from utils.data_utils import one_hot_encode
        y_train_onehot = one_hot_encode(y_train, self.output_size)
        
        # Initialize history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Number of samples and batches
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # Training
        start_time = time.time()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_onehot[indices]
            
            # Initialize epoch metrics
            epoch_loss = 0.0
            epoch_correct = 0
            
            # Batch training
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                # Get batch data
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                activations, layer_inputs = self.forward(X_batch)
                
                # Calculate loss
                y_pred = activations[-1]
                
                # batch_loss = -np.sum(y_batch * np.log(y_pred + 1e-10)) / X_batch.shape[0]
                batch_loss = self.loss_function(y_pred, y_batch, X_batch)
                # Add L2 regularization to loss
                if self.weight_decay > 0:
                    l2_reg = 0
                    for w in self.weights:
                        l2_reg += np.sum(w ** 2)
                    batch_loss += 0.5 * self.weight_decay * l2_reg / X_batch.shape[0]
                
                # Calculate accuracy
                batch_correct = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
                
                #backward pass
                self.backward(X_batch, y_batch, activations, layer_inputs, optimizer, learning_rate)
                
                #update epoch metrics
                epoch_loss += batch_loss * (end_idx - start_idx)
                epoch_correct += batch_correct
            
            # Calculate epoch metrics
            epoch_loss /= n_samples
            epoch_acc = epoch_correct / n_samples
            
            #Validation 
            val_activations, _ = self.forward(X_val)
            val_y_pred = val_activations[-1]
            val_y_true = one_hot_encode(y_val, self.output_size)
            
            # val_loss = -np.sum(val_y_true * np.log(val_y_pred + 1e-10)) / X_val.shape[0]
            val_loss = self.loss_function(val_y_pred, val_y_true, X_val)
            
            if self.weight_decay > 0:
                l2_reg = 0
                for w in self.weights:
                    l2_reg += np.sum(w ** 2)
                val_loss += 0.5 * self.weight_decay * l2_reg / X_val.shape[0]
                
            val_acc = np.mean(np.argmax(val_y_pred, axis=1) == y_val)
            
            # Update history
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            if verbose and (epoch % 1 == 0 or epoch == epochs - 1):
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': epoch_loss,
                    'train_accuracy': epoch_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'epoch_time': epoch_time
                })
        
        total_time = time.time() - start_time
        if verbose:
            print(f"Training completed in {total_time:.2f}s")
        
        if use_wandb:
            wandb.log({'total_training_time': total_time})
        
        return history
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        - X: Input data
        
        Returns:
        - predictions: Predicted class indices
        """
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)
    
    def evaluate(self, X, y):
        """
        Evaluate the model.
        
        Parameters:
        - X: Input data
        - y: Target labels
        
        Returns:
        - accuracy: Classification accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)