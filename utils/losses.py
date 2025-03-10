import numpy as np

class CrossEntropyLoss:
    def __call__(self, y_pred, y_true, X_batch):
        """
        Compute cross-entropy loss.

        Parameters:
        - y_pred: Predicted probabilities
        - y_true: True labels (one-hot encoded)
        - X_batch: Input batch (used for normalization)

        Returns:
        - Cross-entropy loss value
        """
        loss = -np.sum(y_true * np.log(y_pred + 1e-10)) / X_batch.shape[0]
        return loss

class MeanSquaredError:
    def __call__(self, y_pred, y_true,X_batch):
        """
        Compute mean squared error loss.

        Parameters:
        - y_pred: Predicted values
        - y_true: True values

        Returns:
        - MSE loss value
        """
        return np.mean(np.square(y_pred - y_true))

class BinaryCrossEntropy:
    def __call__(self, y_pred, y_true, X_batch):
        """
        Compute binary cross-entropy loss.

        Parameters:
        - y_pred: Predicted probabilities
        - y_true: True labels (0 or 1)

        Returns:
        - Binary cross-entropy loss value
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Dictionary mapping loss function names to instances
LOSS_FUNCTIONS = {
    'cross_entropy_loss': CrossEntropyLoss(),
    'mean_squared_error': MeanSquaredError(),
    'binary_cross_entropy': BinaryCrossEntropy()
}

def get_loss_function(name):
    """
    Get loss function by name.

    Parameters:
    - name: Name of loss function

    Returns:
    - Loss function instance
    """
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_FUNCTIONS[name]



