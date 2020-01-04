import numpy as np
from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, x_file, y_file, ext, test_size):
        self.x_file = x_file
        self.y_file = y_file
        self.ext = ext
        self.test_size = test_size

    def split_data(self):
        X = self.load_input_data()
        y = self.load_output_data()

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def load_input_data(self):
        return np.load(self.x_file + self.ext)

    def load_output_data(self):
        return np.load(self.y_file + self.ext)

    def save_data(self):
        np.save(self.x_file + "_train" + self.ext, self.X_train)
        np.save(self.y_file + "_train" + self.ext, self.y_train)
        np.save(self.x_file + "_test" + self.ext, self.X_test)
        np.save(self.y_file + "_test" + self.ext, self.y_test)
