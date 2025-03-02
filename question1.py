import wandb
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# num_classes = np.unique(y_train).shape[0]
class_labels = {0: "T-shirt/top",
          1: "Trouser",
          2: "Pullover",
          3: "Dress",
          4: "Coat",
          5: "Sandal",
          6: "Shirt",
          7: "Sneaker",
          8: "Bag",
          9: "Ankle boot"}

wandb.init(project="neural-network-fashion-mnist")
num_classes = len(class_labels)

wandb.log({"num_classes": num_classes}) 

# fig = plt.figure(figsize=(num_classes,1))
# fig, axes = plt.subplots(1, num_classes, figsize=(15, 2))
wandb_images = []
for i in range(num_classes):
    idx = np.random.choice(np.where(y_train == i)[0])  # Select a random image from class i
    image = x_train[idx]  # Get the image
    
    wandb_images.append(wandb.Image(image, caption=class_labels[i]))
# for i in range(num_classes):
#     # idx = np.random.choice(np.where(y_train==i)[0], 1).item()
#     # plt.subplot(1, num_classes, i+1)
#     # plt.imshow(x_train[idx].reshape(28,28), cmap="gray")
#     # plt.xlabel(class_labels[y_train[idx]])
    
#     idx = np.random.choice(np.where(y_train == i)[0])
#     axes[i].imshow(x_train[idx], cmap="gray", interpolation="nearest")
    
#     axes[i].set_xticks([])
#     axes[i].set_yticks([])
#     axes[i].set_title(class_labels[i], fontsize=10)
# plt.tight_layout()
    
# plt.savefig("fmnist.png")

wandb.log({"Fashion MNIST": wandb_images})
wandb.finish()
