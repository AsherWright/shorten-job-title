from data_handling.easy_data_generator import EasyDataGenerator


def generate_raw_data():
    dg = EasyDataGenerator("data/easy_titles.csv")
    dg.generate_simple_data()
    dg.save_data()


if __name__ == '__main__':
    generate_raw_data()
