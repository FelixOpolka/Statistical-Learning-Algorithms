import math
import operator
import importlib.util
import random
import csv

spec = importlib.util.spec_from_file_location("decision_tree", "../decision tree/DecisionTree.py")
dt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dt)
# loader = importlib.machinery.SourceFileLoader("decision_tree",
#                                               "../decision tree/DecisionTree.py")
# dt = loader.exec_module('decision_tree')


def increment_counter(dictionary, key, value=1):
    """Increments the counter for a given key in the given dictionary or
    sets the counter to 1 if it does not yet exist."""
    if key in dictionary:
        dictionary[key] += value
    else:
        dictionary[key] = value


class AdaBoostEnsemble:

    def __init__(self):
        self.weak_learners = []
        self.learner_weights = []
        self.weights = []

    def train(self, data_points, subset_size, ensemble_size, trained_learner,
              evaluate):

        self.weights = [1.0/len(data_points) for _ in range(len(data_points))]

        for index in range(ensemble_size):
            print("Trained learner", index)
            self.__add_weak_learner(data_points, subset_size, trained_learner,
                                    evaluate)

    def evaluate(self, input, evaluate):
        votes = {}
        for weak_learner, weight in zip(self.weak_learners,
                                        self.learner_weights):
            prediction = evaluate(weak_learner, input)
            increment_counter(votes, prediction, weight)
        prediction = max(votes.items(), key=operator.itemgetter(1))[0]
        return prediction

    def __training_subset(self, data_points, size):
        subset_indices = random.choices(range(len(data_points)),
                                        weights=self.weights, k=size)
        # subset_indices = np.random.choice(len(data_points), size,
        #                                   p=self.weights)
        return [data_points[index] for index in subset_indices]

    def __error(self, prediction_results):
        error = 0.0
        for prediction_correct, weight in zip(prediction_results,
                                              self.weights):
            if not prediction_correct:
                error += weight
        return error

    @staticmethod
    def __correction(error):
        correction = error / (1.0 - error)
        if correction == 0:
            correction = 0.0000001
        return correction

    def __update_weights(self, prediction_results, correction):
        for index, prediction_correct in enumerate(prediction_results):
            if prediction_correct:
                self.weights[index] *= correction

        normalisation_const = sum(self.weights)
        for index, prediction_correct in enumerate(prediction_results):
            self.weights[index] /= normalisation_const

    def __add_weak_learner(self, data_points, subset_size, trained_learner,
                           evaluate):
        subset = self.__training_subset(data_points, subset_size)
        weak_learner = trained_learner(subset)
        prediction_results = [evaluate(weak_learner, point) == point[-1]
                              for point in data_points]

        # print(prediction_results)
        error = self.__error(prediction_results)
        if error > 0.5:
            print("Training unsuccessful")
            return
        print(error)

        correction = self.__correction(error)
        self.__update_weights(prediction_results, correction)

        self.weak_learners.append(weak_learner)
        self.learner_weights.append(math.log(1.0 / correction))


def parse(s):
    if "." in s:
        return float(s)
    else:
        return int(s)


def create_learner(subset):
    weak_lerner = dt.ClassificationTree()
    weak_lerner.build(subset, 3)
    return weak_lerner


def wl_evaluate(wl, input):
    return wl.evaluate(input)


with open('../data sets/LetterDataSet.csv') as fd:
    data = [tuple([parse(x) for x in line]) for line in csv.reader(fd)]

random.shuffle(data)

training_set_length = 15000
training_set = data[:training_set_length]
test_set = data[training_set_length:]

ensemble = AdaBoostEnsemble()
ensemble.train(training_set, training_set_length, 80, create_learner, wl_evaluate)

print(ensemble.learner_weights)
correct_count = 0
for point in test_set:
    input = point[0:-1]
    output = point[-1]
    prediction = ensemble.evaluate(input, wl_evaluate)
    if prediction == output:
        correct_count += 1

print("Accuracy:", str(correct_count / len(test_set)))

# with open('../data sets/IrisDataSet.csv') as fd:
#     data = [tuple([parse(x) for x in line]) for line in csv.reader(fd)]
#
# random.shuffle(data)
#
# test_set_length = 130
# test_set = data[test_set_length:]
# training_set = data[:test_set_length]
#
# ensemble = AdaBoostEnsemble()
# ensemble.train(training_set, test_set_length, 30, create_learner, wl_evaluate)
#
# correct_count = 0
# for point in test_set:
#     input = point[0:-1]
#     output = point[-1]
#     prediction = ensemble.evaluate(input, wl_evaluate)
#     if prediction == output:
#         correct_count += 1
#
# print("Accuracy:", str(correct_count / len(test_set)))
