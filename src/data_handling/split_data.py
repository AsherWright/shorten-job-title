from data_handling.data_splitter import DataSplitter


def split_data():
    data_splitter = DataSplitter(
        "data/simple_titles_X",
        "data/simple_titles_y",
        ".npy",
        0.3
    )

    data_splitter.split_data()
    data_splitter.save_data()


if __name__ == '__main__':
    split_data()
