import pandas as pd
import json
from data_handling.easy_data_generator import EasyDataGenerator
from data_handling.med_data_generator import MedDataGenerator
from data_handling.hard_data_generator import HardDataGenerator
from data_handling.seq_data_formatter import SeqDataFormatter
from data_handling.data_splitter import DataSplitter


# TODO: this needs to be cleaned up a lot
def handle_data():
    generate_raw_data()
    sequentially_format_data("data/easy_titles.pickle", "data/easy_titles")
    sequentially_format_data("data/med_titles.pickle", "data/med_titles")
    sequentially_format_data("data/hard_titles.pickle", "data/hard_titles")
    split_data("data/easy_titles_X", "data/easy_titles_y", ".npy")
    split_data("data/med_titles_X", "data/med_titles_y", ".npy")
    split_data("data/hard_titles_X", "data/hard_titles_y", ".npy")


def generate_raw_data():
    easy_data_file = "data/easy_titles.pickle"
    med_data_file = "data/med_titles.pickle"
    hard_data_file = "data/hard_titles.pickle"

    easy_dg = EasyDataGenerator(easy_data_file)
    med_dg = MedDataGenerator(med_data_file)
    hard_dg = HardDataGenerator(hard_data_file)

    easy_dg.generate_easy_data()
    med_dg.generate_med_data()
    hard_dg.generate_hard_data()

    easy_dg.save_data()
    med_dg.save_data()
    hard_dg.save_data()


def sequentially_format_data(input_path, output_path):
    glove_vectors = get_glove_vectors()
    titles = pd.read_pickle(input_path)
    WORDS_PER_LONG_TITLE = 10

    seq_data_formatter = SeqDataFormatter(
        titles,
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


def split_data(x_file, y_file, ext):
    data_splitter = DataSplitter(x_file, y_file, ext, 0.3)

    data_splitter.split_data()
    data_splitter.save_data()


if __name__ == '__main__':
    handle_data()
