import numpy as np
import matplotlib.pyplot as plt
import time
import wandb
import argparse
from models.neural_network import FeedforwardNeuralNetwork
from utils.data_utils import preprocess_data
# from visualization.visualize import plot_predictions

def main():
    # Parse command line arguments for hyperparameters
    parser = argparse.ArgumentParser(description='Train a feedforward neural network on Fashion-MNIST')
    parser.add_argument('--hidden_layers', type=str, default='128,64', help='Hidden layer sizes separated by commas')
    parser.add_argument('--activation', type=str, default='tanh', help='Activation function: relu or sigmoid or tanh')
    parser.add_argument('--optimizer', type=str, default='momentum', help='Optimizer: sgd, momentum, nesterov, rmsprop, adam, nadam')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    args = parser.parse_args()
    
    # Convert hidden_layers string to list of integers
    hidden_layers = [int(size) for size in args.hidden_layers.split(',')]
    
    # Initialize wandb
    run = wandb.init(
        project="neural-network-fashion-mnist",
        config={
            "hidden_layers": hidden_layers,
            "activation": args.activation,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        }
    )
    
    # Fashion-MNIST class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Log class names as artifact
    class_names_artifact = wandb.Artifact("class_names", type="dataset")
    with class_names_artifact.new_file("class_names.txt") as f:
        for name in class_names:
            f.write(f"{name}\n")
    wandb.log_artifact(class_names_artifact)
    
    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data()
    
    # Print dataset shapes
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    # Define model architecture
    input_size = 784  # 28x28 pixels
    output_size = 10  # 10 classes in Fashion-MNIST
    
    # Create and train the model
    model = FeedforwardNeuralNetwork(input_size, hidden_layers, output_size, activation=args.activation)
    optimizers = ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]
    for i in optimizers:
        # Train with the specified optimizer
        print(f"\nTraining with {i}:")
        history = model.train(
            X_train, y_train, X_val, y_val, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            optimizer=args.optimizer, 
            learning_rate=args.learning_rate, 
            verbose=True,
            use_wandb=True  # Enable wandb logging in training
        )
        print(history)
        # Evaluate the model
        test_acc = model.evaluate(X_test, y_test)
        print(f"\nTest accuracy with {args.optimizer}: {test_acc:.4f}")
        
        # Log final test metrics
        wandb.log({"test_accuracy": test_acc})
        
        # Make some predictions
        sample_indices = np.random.choice(len(X_test), 5)
        sample_images = X_test[sample_indices]
        sample_labels = y_test[sample_indices]
        predictions = model.predict(sample_images)
        
        # Log sample predictions to wandb
        fig = plt.figure(figsize=(12, 3))
        for i, idx in enumerate(sample_indices):
            plt.subplot(1, 5, i + 1)
            plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[predictions[i]]}")
            plt.axis('off')
        plt.tight_layout()
        wandb.log({"sample_predictions": wandb.Image(fig)})
        plt.close(fig)
        
        # Create a confusion matrix
        from sklearn.metrics import confusion_matrix
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
    
    # Save model as artifact
    # model_artifact = wandb.Artifact("fashion_mnist_model", type="model")
    # model.save("model.pkl")
    # model_artifact.add_file("model.pkl")
    # wandb.log_artifact(model_artifact)
    
    wandb.finish()

if __name__ == "__main__":
    main()