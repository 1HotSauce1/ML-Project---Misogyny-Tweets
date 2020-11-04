import csv
import string
from emoji import UNICODE_EMOJI
from nltk.tokenize import TweetTokenizer
from collections import Counter
from sklearn.preprocessing import normalize


# 1. Read from csv file.
# 2.1 Extract the words from the strings without the delimitators (ex: .,:;'").
# 2.2 Get the labels from the raw data.
# 3. Convert from string array to dictionary.

# 1.
def read_data(csv_file):
    """
    Function that reads data from csv files.
    :param csv_file: Name of the csv file that contains the data
    :return: data_set(list) A list containing all the data from the file
    """
    data_set = []
    with open(csv_file, newline='', encoding='utf-8') as file:
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
def tweet_filter(word):
    """
    Function that filters strings by getting rid of: hashtags, links, punctuation marks,
        numbers, emojis and words with 3 or more of the same letter in a row (italian grammar)
    :param word: Word that needs to be filtered
    :return: True/False(boolean) -> If the word passes or not the filter
    """
    if ('https' not in word and '#' not in word and '@' not in word and is_number(word) is not True
            and word not in list(string.printable) and word not in UNICODE_EMOJI and len(word) > 3):
        return True
    return False


def tokenize(data_set):
    tknzr = TweetTokenizer(reduce_len=True)
    tokens = [tknzr.tokenize(x[1].lower()) for x in data_set]

    for i in range(0, len(tokens)):
        tokens[i] = [x for x in tokens[i] if tweet_filter(x)]

    return tokens


def get_labels(data_set):
    """
    Function that gets the labels from prereaded data from a csv file
    :param data_set: Data from a csv file
    :return: labels(list) The labels that tell if a tweet is mosogynistic or not
    """
    labels = [x[2] for x in data_set]
    return labels


# 3.
def get_corpus_vocabulary(tokens):
    counter = Counter()
    for tok in tokens:
        counter.update(set(tok))

    return counter.most_common(250)


def corpus_to_bow(tokens, corpus):
    corpus = dict(corpus)
    bow = [dict.fromkeys(corpus, 0) for i in range(0, len(tokens))]

    for i in range(0, len(tokens)):
        for j in tokens[i]:
            if j in bow[i]:
                bow[i][j] += 1
    return bow


def bow_to_frequency(bow):
    frequency = [list(x.values()) for x in bow]
    return frequency


# Absolute sum of elements
def abs_sum(frequency):
    linearized = []
    for elem in frequency:
        linearized.append([x / sum(elem) if x > 0 else 0 for x in elem])
    return linearized


def euclid_lin(frequency):
    linearized = []
    for elem in frequency:
        linearized.append([x / euclid_norm(elem) if x > 0 else 0 for x in elem])
    return linearized


def euclid_norm(v):
    suma = 0
    for i in v:
        suma += i ** 2
    return suma


#####
def write_to_csv(labels, file_name):
    with open(file_name, 'w', newline='') as csvfile:
        fieldnames = ['id', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        idx = 5001
        for i in labels:
            writer.writerow({'id': idx, 'label': i})
            idx += 1
