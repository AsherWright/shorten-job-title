from data_handling.generate_raw_data import generate_raw_data
from data_handling.sequentially_format_data import sequentially_format_data
from data_handling.split_data import split_data


def handle_data():
    generate_raw_data()
    sequentially_format_data()
    split_data()


if __name__ == '__main__':
    handle_data()
