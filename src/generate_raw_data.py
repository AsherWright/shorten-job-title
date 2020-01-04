from data_generator import DataGenerator

if __name__ == '__main__':
    dg = DataGenerator("data/simple_titles.csv")
    dg.generate_simple_data()
    dg.save_data()
