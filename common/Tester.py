"""Tester for various predictors. Given a data set file (csv)
the tester loads the data sets, separates it into training and
test set and performs the training and subsequent testing."""

# Currently, only classification tests supported

import csv
from random import shuffle


def __parse(string):
    """Parses a given string into a float if it contains a dot, into
    integer otherwise.
    :param string: Given string to parse.
    :return: Integer or float representation of the given string. """
    if "." in string:
        return float(string)
    return int(string)


def __load_data_set(path, test_set_length):
    """Loads the data set specified by its path (pointing to a csv file),
    shuffles it and returns the resulting training and test set.
    :param path: Path pointing to the data set's csv file. Output variable
    must be the last column.
    :param test_set_length: Length of the resulting test set in number
    of tuples. The rest of the data set will make up the training set.
    :return: Training and test set resulting from data set."""
    with open(path) as file:
        data = [tuple([__parse(x) for x in line]) for line in csv.reader(file)]
    shuffle(data)
    test_set = data[:test_set_length]
    training_set = data[test_set_length:]
    return training_set, test_set


def __get_accuracy(predictor, test_set, evaluate):
    """Calculates the accuracy of a given classification predictor using
    the given test set.
    :param predictor: Predictor to test.
    :param test_set: Test set to use for testing.
    :param evaluate: Function that is used to evaluate the predictor.
    Should take as arguments the predictor to evaluate and the input
    and returns corresponding prediction.
    :return: Measured accuracy of the predictor."""
    correct_count = 0
    for point in test_set:
        input = point[0:-1]
        output = point[-1]
        prediction = evaluate(predictor, input)
        if prediction == output:
            correct_count += 1
    return correct_count/len(test_set)


def test(path, test_set_length, train, evaluate):
    """Tests a classification predictor on a given data sets and
    returns its accuracy.
    :param data-set: Path pointing to the data set's csv file. Output
    variable must be the last column. Output must be discrete variable.
    :param test_set_length: Length of the test set constructed from the
    data set in tuples. The rest of the data set will make up the training
    set.
    :param train: Method for training the predictor that should be
    tested. Given a training set should return a trained predictor.
    :param evaluate: Method for evaluating the predictor that should be
    tested. Given a predictor and an input should return corresponding
    prediction.
    :return: Measured accuracy of the predictor."""
    training_set, test_set = __load_data_set(path, test_set_length)
    predictor = train(training_set)
    return __get_accuracy(predictor, test_set, evaluate)
