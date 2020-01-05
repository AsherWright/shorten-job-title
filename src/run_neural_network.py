from models.neural_network import NeuralNetwork
import numpy as np


def run_neural_network():
    X_train = np.load("data/simple_titles_X_train.npy")
    X_test = np.load("data/simple_titles_X_test.npy")
    y_train = np.load("data/simple_titles_y_train.npy")
    y_test = np.load("data/simple_titles_y_test.npy")

    X_train, X_test = map(lambda x: flatten_data(x), [X_train, X_test])

    neural_net = NeuralNetwork(500, 10, 2, 20)
    neural_net.train(X_train, y_train, 100, 50, 0.3)

    print(X_test[0])
    print(neural_net.predict(X_test[0].reshape(1, 500)))
    loss, acc = neural_net.evaluate(X_test, y_test)

    print("Test sample accuracy = " + str(acc))


def flatten_data(data):
    new_data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))

    return new_data


if __name__ == '__main__':
    run_neural_network()
