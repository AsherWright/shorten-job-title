from data_handling.data_generator import DataGenerator


def generate_raw_data():
    dg = DataGenerator("data/simple_titles.csv")
    dg.generate_simple_data()
    dg.save_data()


if __name__ == '__main__':
    generate_raw_data()
