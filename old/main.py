import wandb
from keras.datasets import fashion_mnist
from nn import NeuralNetwork
from optimizers import SGD, Momentum, NAG, RMSprop, Adam, Nadam
import numpy as np
from utils import one_hot_encode, preprocess_data

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test = preprocess_data(X_train, X_test)  # Normalize and flatten into flat 1D vectors of size 784 (28 × 28).

y_train = one_hot_encode(y_train)   # Convert labels to one-hot vectors
y_test = one_hot_encode(y_test)   # Convert labels to one-hot vectors

# Define class labels
class_labels = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
                5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}

# Initialize WandB
wandb.init(project="BackPropagation")
# wandb.log({"num_classes": len(class_labels)})
# print(len(class_labels))
# print(X_train[0].shape)

# Create model
nn = NeuralNetwork(input_dim=784, output_dim=10, hidden_layers=[128, 64], activation="relu")

# Train model
optimizer = SGD()  # Change optimizer as needed
# nn.train(X_train, y_train, optimizer=optimizer, epochs=10, batch_size=32, x_val=X_test, y_val=y_test)

wandb.finish()