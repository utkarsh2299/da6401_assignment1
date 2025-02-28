import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.lr * grad

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr, self.momentum = lr, momentum
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(param) for param in params]
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grad
            param += self.velocity[i]

class NAG(Momentum):
    def update(self, params, grads):
        for i, (param, grad) in enumerate(zip(params, grads)):
            lookahead = param + self.momentum * self.velocity[i]
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grad
            param += self.velocity[i]

class RMSprop:
    def __init__(self, lr=0.001, decay=0.9, epsilon=1e-8):
        self.lr, self.decay, self.epsilon = lr, decay, epsilon
        self.sq_grads = None

    def update(self, params, grads):
        if self.sq_grads is None:
            self.sq_grads = [np.zeros_like(param) for param in params]
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.sq_grads[i] = self.decay * self.sq_grads[i] + (1 - self.decay) * grad**2
            param -= self.lr * grad / (np.sqrt(self.sq_grads[i]) + self.epsilon)

class Adam(RMSprop):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr, beta2, epsilon)
        self.m = None
        self.beta1=beta1

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            super().update(params, [self.m[i]])

class Nadam(Adam):
    pass  # Can be improved by adding Nesterov gradient modification
