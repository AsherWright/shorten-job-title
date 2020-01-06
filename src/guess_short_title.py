import json
import numpy as np
import pandas as pd
from short_title_guesser import ShortTitleGuesser
from pred_models.simple_rnn import SimpleRnn
from data_handling.seq_data_formatter import SeqDataFormatter


def guess_short_title():
    model = get_simple_rnn_model()
    data_formatter = get_seq_data_formatter()
    long_title = "sales sales head of marketing sales sales"

    stg = ShortTitleGuesser(model, data_formatter)
    print(stg.guess_short_title(long_title))


# TODO: Should instead load an existing model
def get_simple_rnn_model():
    X_train = np.load("data/med_titles_X_train.npy")
    y_train = np.load("data/med_titles_y_train.npy")

    simple_rnn = SimpleRnn(50, 10, 0.1, 0.1)
    simple_rnn.train(X_train, y_train, 100, 50, 0.3)

    return simple_rnn.get_model()


# TODO: Shouldn't need to make an entire data formatter here.
def get_seq_data_formatter():
    glove_vectors = get_glove_vectors()
    med_titles = pd.read_csv("data/med_titles.csv")
    output_path = "data/med_titles"
    WORDS_PER_LONG_TITLE = 10

    return SeqDataFormatter(
        med_titles,
        glove_vectors,
        WORDS_PER_LONG_TITLE,
        output_path
    )


def get_glove_vectors():
    with open('data/embeddings/glove.json') as f:
        data = json.load(f)

    return data


if __name__ == '__main__':
    guess_short_title()
