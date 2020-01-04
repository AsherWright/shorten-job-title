import numpy as np


class DataFormatter:
    def __init__(self, raw_data, embeddings, words_per_title, output_path):
        self.raw_data = raw_data
        self.output_path = output_path
        self.embeddings = embeddings

        self.embedding_length = len(next(iter(self.embeddings.values())))
        self.title_word_count = words_per_title
        self.data_row_count = len(raw_data.index)

    def format_data(self):
        formatted_input_data = []
        formatted_output_data = []

        for index, row in self.raw_data.iterrows():
            formatted_input_data.append(self.format_row_input(row))
            formatted_output_data.append(self.format_row_output(row))

        self.formatted_input_data = np.array(formatted_input_data)
        self.formatted_output_data = np.array(formatted_output_data)

    def format_row_input(self, row):
        formatted_row = []
        long_title = row['long_title']
        words = long_title.split(' ')

        for i in range(self.title_word_count):
            if (i < len(words)):
                formatted_row.append(self.get_embedding_for_word(words[i]))
            else:
                formatted_row.append(np.zeros(self.embedding_length))
        return formatted_row

    def format_row_output(self, row):
        short_title_index = row['short_title_index']

        output = np.zeros(self.title_word_count)
        output[short_title_index] = 1

        return output

    def save_data(self):
        np.save(self.output_path + "_input", self.formatted_input_data)
        np.save(self.output_path + "_output", self.formatted_output_data)

    def get_embedding_for_word(self, word) -> str:
        return self.embeddings[word]
