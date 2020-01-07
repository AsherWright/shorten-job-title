import numpy as np
import pandas as pd


# TODO: Should probably be merged or share code with easy
class MedDataGenerator:
    def __init__(self, save_path):
        self.save_path = save_path

    def generate_med_data(self):
        short_titles = []
        long_titles = []
        short_title_indices = []

        for pre_padding in range(5):
            for post_padding in range(5):
                for _i in range(100):
                    lt, st, sti = get_long_short_titles(
                        pre_padding,
                        post_padding
                    )
                    long_titles.append(lt)
                    short_titles.append(st)
                    short_title_indices.append([sti])

        self.long_titles = long_titles
        self.short_titles = short_titles
        self.short_title_indices = short_title_indices

    def save_data(self):
        data = {
            'long_title': self.long_titles,
            'short_title': self.short_titles,
            'short_title_indices': self.short_title_indices
        }

        pd.DataFrame(data).to_pickle(self.save_path)


def get_long_short_titles(pre_padding, post_padding):
    words = [
        "marketing",
        "sales",
        "ceo",
        "chief",
        "finance",
        "north",
    ]

    short_title = np.random.choice(words)
    prefix = get_prefix(pre_padding, words)
    suffix = get_suffix(post_padding, words)

    long_title = prefix + "head of " + short_title + suffix
    short_title_index = pre_padding + 2

    return long_title, short_title, short_title_index


def get_prefix(pre_padding, words):
    prefix = ""

    for i in range(pre_padding):
        prefix += np.random.choice(words) + " "

    return prefix


def get_suffix(post_padding, words):
    if post_padding == 0:
        return ""

    suffix = ""

    for i in range(post_padding):
        suffix += " " + np.random.choice(words)

    return suffix
