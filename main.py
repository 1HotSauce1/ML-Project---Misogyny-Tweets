import convert_text as ct
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import normalize
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
    lin_train = normalize(freq, axis=1)

    # ------------Testing Data------------ #
    test_data = ct.read_data(test_file)
    tokens = ct.tokenize(test_data)
    corpus_vocabulary = ct.get_corpus_vocabulary(tokens)
    bow = ct.corpus_to_bow(tokens, corpus_vocabulary)
    freq = ct.bow_to_frequency(bow)
    lin_test = normalize(freq, axis=1)

    # ------------KNN Algorithm------------ #
    # k = 5
    # neigh = KNeighborsClassifier(n_neighbors=k, algorithm='brute', metric='euclidean')
    # neigh.fit(lin_train, train_labels)
    # test_labels = neigh.predict(lin_test)
    # ct.write_to_csv(test_labels, submission)

    # -----------Naive Bayes Alg----------- #

    for i in range(0, 26):
        X_train, X_test, y_train, y_test = train_test_split(lin_train, train_labels, test_size=0.4)
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        preds = gnb.predict(X_test)
        print("Accuracy: ", metrics.accuracy_score(y_test, preds) * 100)

    gnb = GaussianNB()
    gnb.fit(lin_train, train_labels)
    test_labels = gnb.predict(lin_test)
    ct.write_to_csv(test_labels, submission)

