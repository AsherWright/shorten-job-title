from generate_raw_data import generate_raw_data
from format_data import format_data
from split_data import split_data


def handle_data():
    generate_raw_data()
    format_data()
    split_data()


if __name__ == '__main__':
    handle_data()
