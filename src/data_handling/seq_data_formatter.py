import numpy as np
from data_formatter import DataFormatter


class SeqDataFormatter(DataFormatter):
    def format_row_input(self, row):
        return self.format_long_title(row['long_title'])

    def format_long_title(self, long_title):
        fmt_long_title = []
        words = long_title.split(' ')

        for i in range(self.title_word_count):
            if (i < len(words)):
                fmt_long_title.append(self.get_embedding_for_word(words[i]))
            else:
                fmt_long_title.append(np.zeros(self.embedding_length))

        return fmt_long_title

    def format_row_output(self, row):
        short_title_indices = row['short_title_indices']

        output = np.zeros(self.title_word_count)

        for short_title_index in short_title_indices:
            output[short_title_index] = 1

        return output
