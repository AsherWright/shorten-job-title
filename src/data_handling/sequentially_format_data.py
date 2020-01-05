import pandas as pd
import json
from data_handling.seq_data_formatter import SeqDataFormatter


def sequentially_format_data():
    glove_vectors = get_glove_vectors()
    simple_titles = pd.read_csv("data/simple_titles.csv")
    output_path = "data/simple_titles"
    WORDS_PER_LONG_TITLE = 10

    seq_data_formatter = SeqDataFormatter(
        simple_titles,
        glove_vectors,
        WORDS_PER_LONG_TITLE,
        output_path
    )

    seq_data_formatter.format_data()
    seq_data_formatter.save_data()


def get_glove_vectors():
    with open('data/embeddings/glove.json') as f:
        data = json.load(f)

    return data


if __name__ == '__main__':
    sequentially_format_data()
