import csv
import numpy as np
import string
from emoji import UNICODE_EMOJI


# 1. Read from csv file.
# 2.1 Extract the words from the strings without the delimitators (ex: .,:;'").
# 2.2 Get the labels from the raw data.
# 3. Convert from string array to dictionary.

# 1.
def read_data(cvs_file):
    data_set = []
    with open(cvs_file, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data_set.append(row)
    data_set.pop(0)
    return data_set


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


# 2.
# Get rid of:
# - name handles (start with @)
# - hashtags
# - links (start with https)
# - punctuation marks
# - numbers
# - emojis
# - words with 3 or less letters
def tweet_filter(word):
    if ('https' not in word and '@' not in word and '#' not in word and is_number(word) is not True
            and word not in list(string.printable) and word not in UNICODE_EMOJI and len(word) > 3):
        return True
    return False


def filter_data(data_set):
    string_list = [x[1].split() for x in data_set]
    translator = str.maketrans('', '', string.punctuation)

    for i in range(0, len(string_list)):
        string_list[i] = [x.translate(translator).lower() for x in string_list[i] if tweet_filter(x)]
    return string_list


def get_labels(data_set):
    labels = [x[2] for x in data_set]
    return labels


# 3.
def lists_union(string_data):
    union = list(set(string_data[0]) | set(string_data[1]))
    for i in range(2, len(string_data)):
        union = list(set(union) | set(string_data[i]))
    return union


def dictionary(string_data, union):
    dictionar = []
    for i in range(0, len(string_data)):
        dictionar.append(dict.fromkeys(union, 0))
        for j in range(0, len(string_data[i])):
            if string_data[i][j] in dictionar[i]:
                return True
    return dictionar
