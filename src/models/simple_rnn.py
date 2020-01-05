from model import Model
from keras.models import Sequential
from keras.layers import LSTM, Dense


class SimpleRnn(Model):
    def __init__(
        self,
        embedding_length: int,
        title_word_count: int,
        dropout: float,
        r_dropout: float
    ) -> None:
        self.embedding_length: int = embedding_length
        self.title_word_count: int = title_word_count
        self.dropout: float = dropout
        self.r_dropout: float = r_dropout
        self.name = self.get_model_name()
        self.model = self.load_or_create_model()

    def load_or_create_model(self) -> Sequential:
        if self.model_exists(self.name):
            return self.load_model(self.name)
        else:
            return self.create_model()

    def create_model(self) -> Sequential:
        model = Sequential()

        model.add(
            LSTM(
                self.embedding_length,
                return_sequences=False,
                dropout=self.dropout,
                recurrent_dropout=self.r_dropout
            )
        )

        # output layer (is this cheating?)
        model.add(Dense(self.title_word_count, activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(
        self,
        X_train,
        y_train,
        batch_size,
        epochs,
        validation_split
    ):
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split
        )

        return history

    def predict(self, new_data):
        return self.model.predict(new_data)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def save_model(self):
        return

    def get_model_name(self) -> str:
        base = "simple_rnn"
        emb = "_e" + str(self.embedding_length)
        twc = "_t" + str(self.title_word_count)
        drp = "_d" + str(self.dropout * 100)
        rdp = "_rd" + str(self.r_dropout * 100)

        return base + emb + twc + drp + rdp
