import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense


def train_rnn():
    x_train_file = "data/simple_titles_X_train.npy"
    y_train_file = "data/simple_titles_y_train.npy"

    X = np.load(x_train_file)
    y = np.load(y_train_file)

    model = get_model(50, 10)

    history = model.fit(
        X,
        y,
        batch_size=2048,
        epochs=150,
        validation_split=0.3
    )

    return history


def get_model(embedding_length, title_word_count):
    model = Sequential()

    model.add(
        LSTM(
            embedding_length,
            return_sequences=False,
            dropout=0.1,
            recurrent_dropout=0.1
        )
    )

    # output layer (is this cheating?)
    model.add(Dense(title_word_count, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    train_rnn()
