

import numpy as np

def sgd(weights, biases, delta_weights, delta_biases, optimization_params, learning_rate=0.01, **kwargs):
    """Standard stochastic gradient descent update."""
    for i in range(len(weights)):
        weights[i] -= learning_rate * delta_weights[i]
        biases[i] -= learning_rate * delta_biases[i]

def momentum(weights, biases, delta_weights, delta_biases, optimization_params, learning_rate=0.01, momentum=0.9, **kwargs):
    """Momentum-based gradient descent update.
    vt​=βv_t−1​+(1−β)∇L
    """
    if 'velocity_w' not in optimization_params:
        optimization_params['velocity_w'] = [np.zeros_like(w) for w in weights]
        optimization_params['velocity_b'] = [np.zeros_like(b) for b in biases]
    
    for i in range(len(weights)):
        optimization_params['velocity_w'][i] = momentum * optimization_params['velocity_w'][i] - learning_rate * delta_weights[i]
        optimization_params['velocity_b'][i] = momentum * optimization_params['velocity_b'][i] - learning_rate * delta_biases[i]
        
        weights[i] += optimization_params['velocity_w'][i]
        biases[i] += optimization_params['velocity_b'][i]

def nesterov(weights, biases, delta_weights, delta_biases, optimization_params, learning_rate=0.01, momentum=0.9, **kwargs):
    """Nesterov accelerated gradient descent update."""
    if 'velocity_w' not in optimization_params:
        optimization_params['velocity_w'] = [np.zeros_like(w) for w in weights]
        optimization_params['velocity_b'] = [np.zeros_like(b) for b in biases]
    
    for i in range(len(weights)):
        old_velocity_w = optimization_params['velocity_w'][i].copy()
        old_velocity_b = optimization_params['velocity_b'][i].copy()
        
        optimization_params['velocity_w'][i] = momentum * optimization_params['velocity_w'][i] - learning_rate * delta_weights[i]
        optimization_params['velocity_b'][i] = momentum * optimization_params['velocity_b'][i] - learning_rate * delta_biases[i]
        
        # Apply Nesterov update
        weights[i] += -momentum * old_velocity_w + (1 + momentum) * optimization_params['velocity_w'][i]
        biases[i] += -momentum * old_velocity_b + (1 + momentum) * optimization_params['velocity_b'][i]

def rmsprop(weights, biases, delta_weights, delta_biases, optimization_params, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8, **kwargs):
    """RMSprop update rule.
        st​=βs_t−1​+(1−β)(∇L)^2"""
    if 'cache_w' not in optimization_params:
        optimization_params['cache_w'] = [np.zeros_like(w) for w in weights]
        optimization_params['cache_b'] = [np.zeros_like(b) for b in biases]
    
    for i in range(len(weights)):
        # Update cache with squared gradients
        optimization_params['cache_w'][i] = decay_rate * optimization_params['cache_w'][i] + (1 - decay_rate) * np.square(delta_weights[i])
        optimization_params['cache_b'][i] = decay_rate * optimization_params['cache_b'][i] + (1 - decay_rate) * np.square(delta_biases[i])
        
        # RMSprop update
        weights[i] -= learning_rate * delta_weights[i] / (np.sqrt(optimization_params['cache_w'][i]) + epsilon)
        biases[i] -= learning_rate * delta_biases[i] / (np.sqrt(optimization_params['cache_b'][i]) + epsilon)

def adam(weights, biases, delta_weights, delta_biases, optimization_params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
    """Adam optimizer update rule.
    mt​=β1​m_t−1​+(1−β1​)∇L
    vt​=β2​v_t−1​+(1−β2​)(∇L)^2
    """
    if 't' not in optimization_params:
        optimization_params['t'] = 0
        optimization_params['m_w'] = [np.zeros_like(w) for w in weights]
        optimization_params['m_b'] = [np.zeros_like(b) for b in biases]
        optimization_params['v_w'] = [np.zeros_like(w) for w in weights]
        optimization_params['v_b'] = [np.zeros_like(b) for b in biases]
    
    optimization_params['t'] += 1
    t = optimization_params['t']
    
    for i in range(len(weights)):
        # Update biased first moment estimate
        optimization_params['m_w'][i] = beta1 * optimization_params['m_w'][i] + (1 - beta1) * delta_weights[i]
        optimization_params['m_b'][i] = beta1 * optimization_params['m_b'][i] + (1 - beta1) * delta_biases[i]
        
        # Update biased second raw moment estimate
        optimization_params['v_w'][i] = beta2 * optimization_params['v_w'][i] + (1 - beta2) * np.square(delta_weights[i])
        optimization_params['v_b'][i] = beta2 * optimization_params['v_b'][i] + (1 - beta2) * np.square(delta_biases[i])
        
        # Compute bias-corrected first moment estimate
        m_w_corrected = optimization_params['m_w'][i] / (1 - beta1**t)
        m_b_corrected = optimization_params['m_b'][i] / (1 - beta1**t)
        
        # Compute bias-corrected second raw moment estimate
        v_w_corrected = optimization_params['v_w'][i] / (1 - beta2**t)
        v_b_corrected = optimization_params['v_b'][i] / (1 - beta2**t)
        
        # Adam update
        weights[i] -= learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + epsilon)
        biases[i] -= learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + epsilon)

def nadam(weights, biases, delta_weights, delta_biases, optimization_params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
    """Nadam (Nesterov-accelerated Adam) optimizer update rule."""
    if 't' not in optimization_params:
        optimization_params['t'] = 0
        optimization_params['m_w'] = [np.zeros_like(w) for w in weights]
        optimization_params['m_b'] = [np.zeros_like(b) for b in biases]
        optimization_params['v_w'] = [np.zeros_like(w) for w in weights]
        optimization_params['v_b'] = [np.zeros_like(b) for b in biases]
    
    optimization_params['t'] += 1
    t = optimization_params['t']
    
    for i in range(len(weights)):
        # Update biased first moment estimate
        optimization_params['m_w'][i] = beta1 * optimization_params['m_w'][i] + (1 - beta1) * delta_weights[i]
        optimization_params['m_b'][i] = beta1 * optimization_params['m_b'][i] + (1 - beta1) * delta_biases[i]
        
        # Update biased second raw moment estimate
        optimization_params['v_w'][i] = beta2 * optimization_params['v_w'][i] + (1 - beta2) * np.square(delta_weights[i])
        optimization_params['v_b'][i] = beta2 * optimization_params['v_b'][i] + (1 - beta2) * np.square(delta_biases[i])
        
        # Compute bias-corrected first moment estimate
        m_w_corrected = optimization_params['m_w'][i] / (1 - beta1**t)
        m_b_corrected = optimization_params['m_b'][i] / (1 - beta1**t)
        
        # Compute bias-corrected second raw moment estimate
        v_w_corrected = optimization_params['v_w'][i] / (1 - beta2**t)
        v_b_corrected = optimization_params['v_b'][i] / (1 - beta2**t)
        
        # Calculate the Nesterov momentum term
        m_w_nesterov = beta1 * m_w_corrected + (1 - beta1) * delta_weights[i] / (1 - beta1**t)
        m_b_nesterov = beta1 * m_b_corrected + (1 - beta1) * delta_biases[i] / (1 - beta1**t)
        
        # Nadam update
        weights[i] -= learning_rate * m_w_nesterov / (np.sqrt(v_w_corrected) + epsilon)
        biases[i] -= learning_rate * m_b_nesterov / (np.sqrt(v_b_corrected) + epsilon)

def get_optimizer(optimizer_name):
    """
    Get the optimizer function based on the name.
    
    Parameters:
    - optimizer_name: Name of the optimizer
    
    Returns:
    - optimizer_function: Function to apply the optimizer
    """
    optimizers = {
        'sgd': sgd,
        'momentum': momentum,
        'nesterov': nesterov,
        'rmsprop': rmsprop,
        'adam': adam,
        'nadam': nadam
    }
    
    if optimizer_name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizers[optimizer_name.lower()]