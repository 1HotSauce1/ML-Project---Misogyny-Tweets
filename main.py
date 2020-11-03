import convert_text as ct
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def main():
    pass


if __name__ == '__main__':
    main()
    test_file = 'test.csv'
    train_file = 'train.csv'
    submission = 'submission.csv'

    # -----------Training Data----------- #
    train_data = ct.read_data(train_file)
    train_labels = ct.get_labels(train_data)
    tokens = ct.tokenize(train_data)
    corpus_vocabulary = ct.get_corpus_vocabulary(tokens)
    bow = ct.corpus_to_bow(tokens, corpus_vocabulary)
    freq = ct.bow_to_frequency(bow)
    lin_train = ct.abs_sum(freq)

    # -----------Testing Data----------- #
    test_data = ct.read_data(test_file)
    tokens = ct.tokenize(test_data)
    corpus_vocabulary = ct.get_corpus_vocabulary(tokens)
    bow = ct.corpus_to_bow(tokens, corpus_vocabulary)
    freq = ct.bow_to_frequency(bow)
    lin_test = ct.abs_sum(freq)

    # -----------KNN Algorithm----------- #
    k = 7
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(lin_train, train_labels)
    test_labels = neigh.predict(lin_test)
    ct.write_to_csv(test_labels, submission)




