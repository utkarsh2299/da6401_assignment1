import wandb
# import numpy as np
from models.neural_network_sweep_orig import FeedforwardNeuralNetwork
from utils.data_utils import preprocess_data
from configs.sweep_config import sweep_config  # Import sweep configuration
# print(sweep_config)
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
        
        # model name for tracking
        #hl_{hidden_layers_count}_sz_{hidden_layer_size}_bs_{batch_size}_act_{activation}_opt_{optimizer}_lr_{learning_rate}_wd_{weight_decay}_init_{weight_init}
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
        # from sklearn.metrics import confusion_matrix
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        
        y_pred = model.predict(X_test)
        # cm = confusion_matrix(y_test, y_pred)
        
        # # Plot and log confusion matrix
        # fig, ax = plt.subplots(figsize=(10, 8))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.title('Confusion Matrix')
        # wandb.log({"confusion_matrix": wandb.Image(fig)})
        # plt.close(fig)
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, 
                                                            y_true=y_test, 
                                                            preds=y_pred, 
                                                            class_names=class_names)})
# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="fashion-mnist-hyperparameter-sweep")

# Run the sweep
wandb.agent(sweep_id, train, count=50)  # Run 50 trials