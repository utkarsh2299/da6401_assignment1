

#The sweep configuration are defined here
# Total configuration : 699984

sweep_config = {
    'method': 'bayes',  # Bayesian optimization    # Search Methods: grid, random, bayesian
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': [5,10]
        },
        'hidden_layers_count': {
            'values': [2, 3, 4]
        },
        'hidden_layer_size': {
            'values': [32, 64, 128, 256]
        },
        'weight_decay': {
            'values': [0, 0.0005, 0.005]
        },
        'loss':{
            'values':['cross_entropy_loss']
            # 'values':['mean_squared_error', 'cross_entropy_loss']
        },
        'learning_rate': {
            'values': [1e-1,1e-2,1e-3]
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