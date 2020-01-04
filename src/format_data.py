import pandas as pd
import json
from data_formatter import DataFormatter


def format_data():
    glove_vectors = get_glove_vectors()
    simple_titles = pd.read_csv("data/simple_titles.csv")
    output_path = "data/simple_titles_formatted"
    WORDS_PER_LONG_TITLE = 10

    data_formatter = DataFormatter(
        simple_titles,
        glove_vectors,
        WORDS_PER_LONG_TITLE,
        output_path
    )

    data_formatter.format_data()
    data_formatter.save_data()


def get_glove_vectors():
    with open('data/embeddings/glove.json') as f:
        data = json.load(f)

    return data


if __name__ == '__main__':
    format_data()
