import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import NeuralNetwork
from utils import *
#Basic Dataset
X, y = make_moons(n_samples=200, noise=0.2)
y = y.reshape(-1, 1)


#Learning rate function
def experiment_learning_rate(X, y):
    learning_rates = [0.001, 0.01, 0.1]

    for lr in learning_rates:
        nn = NeuralNetwork(2, 10, 1, lr=lr)
        plot_decision_boundary(nn, X, y)
        losses = nn.train(X, y, epochs=500)
        plt.plot(losses, label=f"lr={lr}")

    plt.title("Learning Rate Comparison")
    plt.legend()
    plt.show()

#Hidden layer size
def experiment_hidden_size(X, y):
    hidden_sizes = [5, 10, 20]

    for h in hidden_sizes:
        nn = NeuralNetwork(2, h, 1, lr=0.01)
        plot_decision_boundary(nn, X, y)
        losses = nn.train(X, y, epochs=500)
        plt.plot(losses, label=f"hidden={h}")

    plt.title("Hidden Layer Size Comparison")
    plt.legend()
    plt.show()

#Epoch Comparison
def experiment_epochs(X, y):
    epochs_list = [100, 500, 1000]

    for ep in epochs_list:
        nn = NeuralNetwork(2, 10, 1, lr=0.01)
        plot_decision_boundary(nn, X, y)
        losses = nn.train(X, y, epochs=ep)
        plt.plot(losses, label=f"epochs={ep}")

    plt.title("Epoch Comparison")
    plt.legend()
    plt.show()


#MNIST DATASET
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.to_numpy() / 255.0  # normalize
    y = mnist.target.astype(int)

    # Binary classification: digit 0 vs not 0
    y = (y == 0).astype(int).to_numpy().reshape(-1, 1)

    return train_test_split(X, y, test_size=0.2, random_state=42)

#MNIST TRAINING

def experiment_mnist():
    X_train, X_test, y_train, y_test = load_mnist()

    nn = NeuralNetwork(784, 64, 1, lr=0.01)
    losses = nn.train(X_train[:5000], y_train[:5000], epochs=50)

    plt.plot(losses)
    plt.title("MNIST Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    # Evaluate
    preds = nn.forward(X_test[:1000])
    preds = (preds > 0.5).astype(int)

    accuracy = (preds == y_test[:1000]).mean()
    print(f"MNIST Accuracy (0 vs not 0): {accuracy * 100:.2f}%")

#calling Function
if __name__ == "__main__":

    #Moons Experiment
    # experiment_learning_rate(X, y)

    #MNIST Dataset
    experiment_mnist()