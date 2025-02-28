import numpy as np
from layers import Dense, Activation
from utils import compute_loss, compute_accuracy
import wandb
class NeuralNetwork:
    def __init__(self, input_dim, output_dim, hidden_layers, activation):
        self.layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            self.layers.append(Dense(prev_dim, h))
            self.layers.append(Activation(activation))
            prev_dim = h
        self.layers.append(Dense(prev_dim, output_dim))  # Output layer
        self.layers.append(Activation("softmax"))  # Softmax for classification
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def train(self, x, y, optimizer, epochs, batch_size, x_val, y_val):
        num_samples = x.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            x_shuffled, y_shuffled = x[indices], y[indices]

            for i in range(0, num_samples, batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward pass
                predictions = self.forward(x_batch)
                loss = compute_loss(y_batch, predictions)

                # Backward pass
                grad = predictions - y_batch
                self.backward(grad)

                # Update weights
                optimizer.update(self.get_params(), self.get_grads())

            # Evaluate on validation set
            val_preds = self.forward(x_val)
            val_loss = compute_loss(y_val, val_preds)
            val_accuracy = compute_accuracy(y_val, val_preds)

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            wandb.log({"epoch": epoch + 1, "loss": loss, "val_loss": val_loss, "val_accuracy": val_accuracy})
    
    def get_params(self):
        return [layer.weights for layer in self.layers if isinstance(layer, Dense)]

    def get_grads(self):
        return [layer.grad_w for layer in self.layers if isinstance(layer, Dense)]
