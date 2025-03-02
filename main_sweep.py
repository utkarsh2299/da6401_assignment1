import wandb
from wandb.keras import WandbCallback
import numpy as np
from keras.datasets import fashion_mnist
import argparse
import os
from models.neural_network import FeedforwardNeuralNetwork
from utils.data_utils import preprocess_data, one_hot_encode

# Define the sweep configuration
sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': [5, 10]
        },
        'hidden_layers_count': {
            'values': [3, 4, 5]
        },
        'hidden_layer_size': {
            'values': [32, 64, 128]
        },
        'weight_decay': {
            'values': [0, 0.0005, 0.5]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'optimizer': {
            'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'weight_init': {
            'values': ['random', 'xavier']
        },
        'activation': {
            'values': ['sigmoid', 'tanh', 'relu']
        }
    }
}

def train():
    # Initialize a new wandb run
    with wandb.init() as run:
        # Access all hyperparameters through wandb.config
        config = wandb.config
        
        # Load and preprocess data
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data()
        
        # Print dataset shapes
        print(f"Training set: {X_train.shape}, {y_train.shape}")
        print(f"Validation set: {X_val.shape}, {y_val.shape}")
        print(f"Test set: {X_test.shape}, {y_test.shape}")
        
        # Define model architecture
        input_size = 784  # 28x28 pixels
        output_size = 10  # 10 classes in Fashion-MNIST
        
        # Create hidden layers configuration based on count and size
        hidden_layers = [config.hidden_layer_size] * config.hidden_layers_count
        
        # Create model name for tracking
        model_name = f"hl_{config.hidden_layers_count}_sz_{config.hidden_layer_size}_bs_{config.batch_size}_act_{config.activation}_opt_{config.optimizer}_lr_{config.learning_rate}_wd_{config.weight_decay}_init_{config.weight_init}"
        wandb.run.name = model_name
        
        # Create the model
        model = FeedforwardNeuralNetwork(
            input_size, 
            hidden_layers, 
            output_size,
            activation=config.activation,
            weight_init=config.weight_init,
            weight_decay=config.weight_decay
        )
        
        # Train the model
        history = model.train(
            X_train, y_train, X_val, y_val,
            epochs=config.epochs,
            batch_size=config.batch_size,
            optimizer=config.optimizer,
            learning_rate=config.learning_rate,
            verbose=True,
            use_wandb=True
        )
        
        # Evaluate on test set
        test_acc = model.evaluate(X_test, y_test)
        wandb.log({"test_accuracy": test_acc})
        
        # Fashion-MNIST class names
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                       
        # Make predictions on test set and create confusion matrix
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot and log confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        wandb.log({"confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="fashion-mnist-hyperparameter-sweep")

# Run the sweep
wandb.agent(sweep_id, train, count=50)  # Run 50 trials