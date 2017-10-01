"""Tests for AdaBoost.py on data sets for decision tree weak learners."""

import importlib.util as il
import AdaBoost as ab

# Load tester module
tester_spec = il.spec_from_file_location("tester", "../common/Tester.py")
tester = il.module_from_spec(tester_spec)
tester_spec.loader.exec_module(tester)

# Load decision tree module
dt_spec = il.spec_from_file_location("dt", "../decision tree/DecisionTree.py")
dt = il.module_from_spec(dt_spec)
dt_spec.loader.exec_module(dt)


def __evaluate_tree(predictor, input):
    """Function for evaluating a given decision tree on a given input."""
    return predictor.evaluate(input)


def __generate_train_func(ensemble_size, depth_constraint):
    """Generates customized train functions for ensembles. Allows to adjust
    training to individual training sets."""
    # Function for training the weak learner (required for AdaBoostEnsemble).
    def __train_tree(training_set):
        tree = dt.ClassificationTree()
        tree.build(training_set, depth_constraint)
        return tree
    # Function for training the ensemble (required for Tester).
    def __train(training_set):
        ensemble = ab.AdaBoostEnsemble()
        ensemble.train(training_set, len(training_set), ensemble_size,
                       __train_tree, __evaluate_tree)
        return ensemble
    return __train


def __evaluate(predictor, input):
    """Function for evaluating a given AdaBoost ensemble on a given input."""
    return predictor.evaluate(input, __evaluate_tree)


def test_iris():
    print("Testing on Iris Data Set...")
    print("Accuracy:", tester.test("../data sets/IrisDataSet.csv", 120,
                                   __generate_train_func(20, 1), __evaluate))


def test_letter():
    print("Testing on Letter Data Set...")
    print("Accuracy:",
          tester.test("../data sets/LetterDataSet.csv", 10000,
                      __generate_train_func(40, 3), __evaluate))


if __name__ == '__main__':
    test_iris()
    test_letter()
