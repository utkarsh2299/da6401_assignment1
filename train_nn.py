# import numpy as np
import wandb
import argparse
from models.neural_network_sweep import FeedforwardNeuralNetwork
from utils.data_utils import preprocess_data
from configs.sweep_config import sweep_config  # Import sweep configuration
# print(sweep_config)
# from visualization.visualize import plot_predictions

def main(args: argparse.Namespace):


    # Convert hidden_layers string to list of integers
    # hidden_layers = [int(size) for size in args.hidden_layers.split(',')]
    
    if args.dataset == "fashion_mnist":
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] # Fashion-MNIST class names
    elif args.dataset == "mnist":
        class_names = ["0", "1", "2", "3", "4","5","6","7","8","9"] # Digit-MNIST class names
        
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(args.dataset)
    
    # Print dataset shapes
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    # Define model architecture
    input_size = 784  # 28x28 pixels
    output_size = 10  # 10 classes in MNIST/ F-MNIST
    
    # print(args.use_wandb)
    
    def train():
        print("inside train")
        with wandb.init() as run:
        # Access all hyperparameters through wandb.config
            config = wandb.config
            # Create hidden layers configuration based on count and size
            hidden_layers = [config.hidden_layer_size] * config.hidden_layers_count
            
            # model name for tracking
            #ep_{epochs}_hl_{hidden_layers_count}_sz_{hidden_layer_size}_bs_{batch_size}_act_{activation}_opt_{optimizer}_lr_{learning_rate}_wd_{weight_decay}_init_{weight_init}
            model_name = f"ep_{config.epochs}_hl_{config.hidden_layers_count}_sz_{config.hidden_layer_size}_bs_{config.batch_size}_act_{config.activation}_opt_{config.optimizer}_lr_{config.learning_rate}_wd_{config.weight_decay}_init_{config.weight_init}"
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
            wandb.log({"Model Run history" : history})
            
            # Evaluate on test set
            test_acc = model.evaluate(X_test, y_test)
            wandb.log({"test_accuracy": test_acc})
            
            y_pred = model.predict(X_test)
            wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, 
                                                            y_true=y_test, 
                                                            preds=y_pred, 
                                                            class_names=class_names)})
            
    if args.use_wandb_sweep == "True": 
        wandb.login()
        # print("In if con")
        # Initialize the sweep
        # sweep_id = wandb.sweep(sweep_config, project="fashion-mnist-hyperparameter-sweep")
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)

        # Run the sweep
        wandb.agent(sweep_id, train, count=100)  # Run 50 trials
        wandb.finish()
        
    else:
        print("In else")
        args.hidden_layers = [args.hidden_layer_size] * args.hidden_layers_count
    # Create and train the model
        model = FeedforwardNeuralNetwork(
                    input_size, 
                    args.hidden_layers, 
                    output_size,
                    activation=args.activation,
                    weight_init=args.weight_init,
                    weight_decay=args.weight_decay,
                    loss=args.loss, 
                    learning_rate=args.learning_rate, 
                    beta1=args.beta1, 
                    beta2=args.beta2, 
                    epsilon=args.epsilon
                )
        
        print(f"\nTraining with {args.optimizer}:")
        history = model.train(
                X_train, y_train, X_val, y_val, 
                epochs=args.epochs, 
                batch_size=args.batch_size, 
                optimizer=args.optimizer, 
                learning_rate=args.learning_rate, 
                verbose=True,
                use_wandb=True 
            )
        print(history)
        # Evaluate the model
        test_acc = model.evaluate(X_test, y_test)
        print(f"\nTest accuracy with {args.optimizer}: {test_acc:.4f}")
               
    
        # wandb.finish()

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="Train a Neural Network with different settings")

    # Arguments for Weights & Biases logging
    parser.add_argument("-uw", "--use_wandb", type=str, default="True",
                        help="Set this to 'true' if you want to track training using Weights & Biases and do the sweep on various hyperparameters")
    parser.add_argument("-uw_s", "--use_wandb_sweep", type=str, default="False",
                        help="Set this to 'True' if you want to do the sweep on various hyperparameters in config")

    parser.add_argument("-wp", "--wandb_project", type=str, default="neural-network-fashion-mnist",
                        help="Name of the WandB project where training details will be logged")

    parser.add_argument("-we", "--wandb_entity", type=str, default="utkarsh",
                        help="Your WandB entity name (kind of like your username for logging experiments)")

    # Dataset selection
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist",
                        help="Pick a dataset for training: ['mnist' or 'fashion_mnist']")

    # Training parameters
    parser.add_argument("-e", "--epochs", type=int, default=1,
                        help="How many times to go through the entire training dataset")

    parser.add_argument("-b", "--batch_size", type=int, default=4,
                        help="How many samples to process at once during training")

    # Loss function
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy_loss",
                        help="Loss function to use: ['mean_squared_error' or 'cross_entropy_loss']")

    # Optimizer settings
    parser.add_argument("-o", "--optimizer", type=str, default="sgd",
                        help="Choose which optimizer to use: ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']")

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1,
                        help="How fast the model should learn (too high may be unstable, too low may be slow)")

    parser.add_argument("-m", "--momentum", type=float, default=0.5,
                        help="Momentum value (only useful if using momentum-based optimizers)")

    parser.add_argument("-beta", "--decay_rate", type=float, default=0.5,
                        help="Beta parameter (only used for RMSprop optimizer)")

    parser.add_argument("-beta1", "--beta1", type=float, default=0.5,
                        help="Beta1 parameter for Adam/Nadam optimizers (helps control moving averages)")

    parser.add_argument("-beta2", "--beta2", type=float, default=0.5,
                        help="Beta2 parameter for Adam/Nadam (controls second moment estimation)")

    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6,
                        help="A tiny number added to avoid division by zero in optimizers")

    # Regularization
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0,
                        help="Weight decay (L2 regularization) to prevent overfitting")

    # Weight initialization
    parser.add_argument("-w_i", "--weight_init", type=str, default="random",
                        help="Choose how to initialize weights: ['random' or 'Xavier']")

    # Network architecture
    parser.add_argument("-n_hl", "--hidden_layers_count", type=int, default=1,
                        help="Number of hidden layers in the neural network")

    parser.add_argument("-sz", "--hidden_layer_size", type=int, default=4,
                        help="Number of neurons in each hidden layer")

    # Activation function
    parser.add_argument("-a", "--activation", type=str, default="sigmoid",
                        help="Activation function to use: ['identity', 'sigmoid', 'tanh', 'ReLU']")

    args = parser.parse_args()
    main(args)