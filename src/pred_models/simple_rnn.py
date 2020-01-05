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

    def get_model_name(self) -> str:
        base = "simple_rnn"
        emb = "_e" + str(self.embedding_length)
        twc = "_t" + str(self.title_word_count)
        drp = "_d" + str(self.dropout * 100)
        rdp = "_rd" + str(self.r_dropout * 100)

        return base + emb + twc + drp + rdp

    def get_model(self) -> Sequential:
        return self.model
