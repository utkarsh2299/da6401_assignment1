# Define the sweep configuration
sweep_config = {
    'method': 'bayes',  # Bayesian optimization    # Search Methods: grid, random, bayesian
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': [2]
        },
        'hidden_layers_count': {
            'values': [2]
        },
        'hidden_layer_size': {
            'values': [32]
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
            'values': ['random', 'xavier', 'he']
        },
        'activation': {
            'values': ['sigmoid', 'tanh', 'relu']
        }
    }
}