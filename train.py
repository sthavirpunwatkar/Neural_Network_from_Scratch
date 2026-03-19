import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from model import NeuralNetwork

# Dataset
X, y = make_moons(n_samples=200, noise=0.2)
y = y.reshape(-1, 1)

# Experiment: learning rate comparison
learning_rates = [0.001, 0.01, 0.1]

for lr in learning_rates:
    nn = NeuralNetwork(2, 10, 1, lr=lr)
    losses = nn.train(X, y, epochs=500)
    plt.plot(losses, label=f"lr={lr}")

plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()