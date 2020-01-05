from pred_models.simple_rnn import SimpleRnn
import numpy as np


def run_simple_rnn():
    X_train = np.load("data/simple_titles_X_train.npy")
    X_test = np.load("data/simple_titles_X_test.npy")
    y_train = np.load("data/simple_titles_y_train.npy")
    y_test = np.load("data/simple_titles_y_test.npy")

    simple_rnn = SimpleRnn(50, 10, 0.1, 0.1)
    simple_rnn.train(X_train, y_train, 100, 50, 0.3)

    print(X_test[0])
    print(simple_rnn.predict(X_test[0].reshape(1, 10, 50)))
    loss, acc = simple_rnn.evaluate(X_test, y_test)

    print("Test sample accuracy = " + str(acc))


if __name__ == '__main__':
    run_simple_rnn()
