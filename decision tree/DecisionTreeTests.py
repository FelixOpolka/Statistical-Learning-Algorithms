"""Tests for DecisionTree.py on data sets."""

import importlib.util
import DecisionTree

spec = importlib.util.spec_from_file_location("tester", "../common/Tester.py")
tester = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tester)

def __train(training_set):
    tree = DecisionTree.ClassificationTree()
    tree.build(training_set)
    return tree


def __evaluate(predictor, input):
    return predictor.evaluate(input)


def test_iris():
    print("Testing on Iris Data Set...")
    print("Accuracy:", tester.test("../data sets/IrisDataSet.csv", 120, __train,
                                   __evaluate))


def test_letter():
    print("Testing on Letter Data Set...")
    print("Accuracy:",
          tester.test("../data sets/LetterDataSet.csv", 10000, __train,
                      __evaluate))


if __name__ == '__main__':
    test_iris()
    test_letter()
