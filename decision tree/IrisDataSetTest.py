import csv
from random import shuffle
import DecisionTree


def parse(s):
    if "." in s:
        return float(s)
    else:
        return int(s)


with open('../data sets/IrisDataSet.csv') as fd:
    data = [tuple([parse(x) for x in line]) for line in csv.reader(fd)]

shuffle(data)

test_set_length = 120
test_set = data[test_set_length:]
training_set = data[:test_set_length]

tree = DecisionTree.ClassificationTree()
tree.build(training_set)
print(tree)

correct_count = 0
for point in test_set:
    input = point[0:-1]
    output = point[-1]
    prediction = tree.evaluate(input)
    if prediction == output:
        correct_count += 1

print("Accuracy:", str(correct_count / len(test_set)))
