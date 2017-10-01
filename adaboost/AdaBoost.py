"""Implementation of an ensemble of classification predictors trained
using the AdaBoost.M1 algorithm."""

import math
import operator
import random


def increment_counter(dictionary, key, value=1):
    """Increments the counter for a given key in the given dictionary or
    sets the counter to 1 if it does not yet exist."""
    if key in dictionary:
        dictionary[key] += value
    else:
        dictionary[key] = value


class AdaBoostEnsemble:
    """An ensemble of classification predictors trained using the
    AdaBoost.M1 algorithm."""

    def __init__(self):
        self.weak_learners = []
        self.learner_weights = []
        self.weights = []

    def train(self, data_points, subset_size, ensemble_size, trained_learner,
              evaluate):
        """Builds the ensemble from the given data set.
        :param data_points: List of tuples representing the data points.
        Last tuple component must corresponds to output label.
        :param subset_size: AdaBoost uses resampled training sets for
        each predictor. Resampled training sets can be smaller than
        original data set. This parameter determines its size in number
        of tuples.
        :param ensemble_size: Number of weak learners in the ensemble.
        :param trained_learner: Function that given a training set returns a
        trained weak learner.
        :param evaluate: Function that given a predictor and an input
        returns the predictor's prediction. """
        # Initialize weights uniformly
        self.weights = [1.0/len(data_points) for _ in range(len(data_points))]
        # Add weak learners one by one
        for _ in range(ensemble_size):
            self.__add_weak_learner(data_points, subset_size, trained_learner,
                                    evaluate)

    def evaluate(self, input, evaluate):
        """Evaluates the ensemble for a given input using majority voting.
        :param input: Input to evaluate.
        :param evaluate: Function that given a weak learner and an input
        returns the weak learner's corresponding prediction.
        :return: Ensemble prediction."""
        votes = {}
        for weak_learner, weight in zip(self.weak_learners,
                                        self.learner_weights):
            prediction = evaluate(weak_learner, input)
            increment_counter(votes, prediction, weight)
        prediction = max(votes.items(), key=operator.itemgetter(1))[0]
        return prediction

    def __training_subset(self, data_points, size):
        """Returns a random set of samples where each sample in the
        entire data set is drawn with a probability determined by its
        value in the weights array."""
        subset_indices = random.choices(range(len(data_points)),
                                        weights=self.weights, k=size)
        return [data_points[index] for index in subset_indices]

    def __error(self, prediction_results):
        """Given the prediction results of a weak learner, returns the
        corresponding error.
        :param prediction_results: Array of boolean values where true denotes
        that the training sample in the entire data set at the
        same index was predicted correctly."""
        error = 0.0
        for prediction_correct, weight in zip(prediction_results,
                                              self.weights):
            if not prediction_correct:
                error += weight
        return error

    @staticmethod
    def __correction(error):
        """Given the error for a weak learner, calculates the corresponding
        correction term."""
        correction = error / (1.0 - error)
        if correction == 0:
            correction = 0.0000001
        return correction

    def __update_weights(self, prediction_results, correction):
        """Updates each the weight of each sample in the entire training
        set if necessary."""
        for index, prediction_correct in enumerate(prediction_results):
            if prediction_correct:
                self.weights[index] *= correction
        # Normalize weights to get a valid probability distribution
        normalisation_const = sum(self.weights)
        for index, prediction_correct in enumerate(prediction_results):
            self.weights[index] /= normalisation_const

    def __add_weak_learner(self, data_points, subset_size, trained_learner,
                           evaluate):
        """Trains a new weak learner, evaluates it, determines its
        weight within the ensemble and updates the entire data set's
        weights."""
        # Create and train weak learner
        subset = self.__training_subset(data_points, subset_size)
        weak_learner = trained_learner(subset)
        # Evaluate weak learner
        prediction_results = [evaluate(weak_learner, point) == point[-1]
                              for point in data_points]
        # Update sample weights based on weak learner's error
        error = self.__error(prediction_results)
        if error > 0.5:
            print("Training unsuccessful")
            return
        correction = self.__correction(error)
        self.__update_weights(prediction_results, correction)
        # Add weak learner to ensemble with corresponding weight
        self.weak_learners.append(weak_learner)
        self.learner_weights.append(math.log(1.0 / correction))
