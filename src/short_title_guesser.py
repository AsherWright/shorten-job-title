import numpy as np
from model import Model
from data_formatter import DataFormatter


class ShortTitleGuesser:
    def __init__(self, model: Model, data_formatter: DataFormatter) -> None:
        self.model = model
        self.data_formatter = data_formatter

    def guess_short_title(self, long_title: str) -> str:
        x = self.data_formatter.format_long_title(long_title)
        x = np.array(x)
        x = x.reshape((1, x.shape[0], x.shape[1]))

        preds = self.model.predict(x)
        print(preds)
        preds = np.round(preds)[0]
        print(preds)
        words = long_title.split(" ")

        short_title_guess = [w for w, p in zip(words, preds) if p == 1]

        return ' '.join(short_title_guess)
