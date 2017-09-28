"""A C4.5 Decision Tree learner for classification problems"""

import math


def increment_counter(dictionary, key):
    """Increments the counter for a given key in the given dictionary or
    sets the counter to 1 if it does not yet exist."""
    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1


def add_to_list(dictionary, key, value):
    """Adds a given value to a list that is specified by its key in the given
    dictionary. If the list does not yet exist, it is created on the fly."""
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]


def partitions_class_attribute(data_points, attr_index):
    """Partitions data points using a given class attribute. Data points
    with the same class label are combined into a partition.
    :param data_points: List of tuples representing the data points.
    :param attr_index: Index of the attribute inside the tuple to be used
    for partitioning."""
    partitioned_data_points = {}
    for point in data_points:
        add_to_list(partitioned_data_points, point[attr_index], point)
    return partitioned_data_points


def partition_sizes_class_attribute(data_points, attribute_index):
    """Returns the number of items in each partition when partitioning
    the given data points using a specified attribute
    :param data_points: The data points which ought to be partitioned
    :param attribute_index: The class-type attribute to use to distinguish
                            the partitions
    :return: A dictionary which maps the given class-attributes labels to
             the number of points with this attribute label"""
    inputs_per_class = {}
    for point in data_points:
        increment_counter(inputs_per_class, point[attribute_index])
    return inputs_per_class


def information(data_points):
    """Returns the average amount of information of the given data_points
    (also known as entropy).
    :param data_points: List of tuples representing the data points.
    Last tuple component corresponds to output label.
    :return: Average amount of information of data_points"""
    inputs_per_class = partition_sizes_class_attribute(data_points, -1)
    entropy = 0.0
    for count in inputs_per_class.values():
        relative_frequency = count / len(data_points)
        information = relative_frequency * math.log2(relative_frequency)
        entropy += (-information)
    return entropy


def information_partitioned(point_count, partitioned_data_points):
    """Returns the average amount of information given the partitioned
    data_points.
    :param point_count: Total number of data_points.
    :param partitioned_data_points: Dictionary of partitions where each
    partition contains a list of tuples representing the data points.
    Last tuple component corresponds to output label."""
    partitioned_entropy = 0.0
    for partition in partitioned_data_points.values():
        partition_size = len(partition)
        entropy = information(partition)
        partitioned_entropy += partition_size / point_count * entropy
    return partitioned_entropy


def split(sorted_data_points, attr_index, split_value):
    """Splits a list of data points sorted by a given element into two
    lists with one list containing tuples <= split_value and one list
    containing tuples > split_value.
    :param sorted_data_points: List of data points sorted by their values
    of the attribute specified by attr_index.
    :param attr_index: Index of the attribute of the tuple used to
    specify order of tuples.
    :param split_value: Value of tuple attribute where list of data points
    is split.
    :return: List containing two lists of data points as specified above."""
    for index, value in enumerate(sorted_data_points):
        if value[attr_index] > split_value:
            return [sorted_data_points[:index], sorted_data_points[index:]]
    return [sorted_data_points, []]


def cont_attr_best_split(data_points, attr_index):
    """Finds the split value for partitioning the data points using
    the given attribute such that the information gain is maximized.
    Searches the partition with minimum information as its minimum maximizes
    information gain (gain = information unpartit. - information partit.).
    :param data_points: List of tuples representing the data points.
    Last tuple component corresponds to output label.
    :param attr_index: Index of the attribute for which the best split
    value is requested.
    :return: The minimum average information when using the optimal split
    value along with the optimal split value."""
    data_points.sort(key=lambda tup: tup[attr_index])
    left_sublist = []
    right_sublist = data_points
    unique_attr_values = list(set([i[attr_index] for i in data_points]))
    unique_attr_values.sort()
    min_info = None
    best_value = None
    for value in unique_attr_values:
        for index in range(len(right_sublist)):
            if right_sublist[index][attr_index] > value:
                left_sublist.extend(right_sublist[:index])
                right_sublist = right_sublist[index:]
                break
        partitioned_data_points = {"l": left_sublist, "r": right_sublist}
        info = information_partitioned(len(data_points), partitioned_data_points)
        if min_info is None or info < min_info:
            min_info = info
            best_value = value
    return min_info, best_value


def cont_attr_gain(data_points, attr_index, info):
    """Calculates the information gain when partitioning the data points
    with the given continuous attribute.
    :param data_points: List of tuples representing the data points.
    Last tuple component corresponds to output label.
    :param attr_index: Index of the attribute used for partitioning
    the data points. Must be continuous attribute type.
    :param info: Average information of the unpartitioned data points.
    :return: Information gain when partitioning with the specified
    attribute along with the best split value for creating the partitioning."""
    min_info, best_value = cont_attr_best_split(data_points, attr_index)
    return info - min_info, best_value


def class_attr_gain(data_points, attr_index, info):
    """Calculates the information gain when partitioning the data points
    with the given class attribute.
    :param: data_points: List of tuples representing the data points.
    Last tuple component corresponds to output label.
    :param attr_index: Index of the attribute used for partitioning
    the data points. Must be class attribute type.
    :param info: Average information of the unpartitioned data points.
    :return: Information gain when partitioning with the specified
    attribute."""
    partitioned_data_points = partitions_class_attribute(data_points, attr_index)
    info_partitioned = information_partitioned(len(data_points), partitioned_data_points)
    return info - info_partitioned


def gain(data_points, attr_index):
    """Calculate the information gain when partitioning the data points
    using the given attribute.
    :param data_points: List of tuples representing the data points.
    Last tuple component corresponds to output label.
    :param attr_index: Index of the attribute used for partitioning
    the data points.
    :return: Information gain achieved when partitioning the data points
    using the given attribute."""
    if len(data_points) == 0:
        return 0.0
    info = information(data_points)
    if type(data_points[0][attr_index]) is int:
        return class_attr_gain(data_points, attr_index, info)
    else:
        return cont_attr_gain(data_points, attr_index, info)[0]


def best_split_attribute(data_points):
    """Determines the attribute which results in the highest information
    gain when used for partitioning the data points.
    :param data_points: List of tuples representing the data points.
    Last tuple component corresponds to output label.
    :return: Index of the attribute with highest information gain along
    with the gain."""
    best_attribute = 0
    best_gain = 0.0
    for attribute_index in range(len(data_points[0])-1):
        curr_gain = gain(data_points, attribute_index)
        if curr_gain > best_gain:
            best_attribute = attribute_index
            best_gain = curr_gain
    return best_attribute, best_gain


class ClassificationTree:
    """A C4.5 decision tree for classification problems."""

    def __init__(self):
        self.root = None

    def build(self, data_points, depth_constraint=None, gain_threshold=0.01):
        """Builds/trains the decision tree using the given data points.
        :param data_points: List of tuples representing the data points.
        Last tuple component must corresponds to output label.
        :param depth_constraint: Specifies the maximum depth of the decision
        tree. A depth constraint of 0 results in a tree with a single
        test (i.e. single inner node). Specifying None means that no depth
        constraint is applied.
        :param gain_threshold: Minimum information gain a test/ inner node
        must achieve to be added to the tree. Otherwise, no further test
        is added to the current branch."""
        self.root = Node.suitable_node(data_points, 0, depth_constraint, gain_threshold)

    def evaluate(self, input):
        """Evaluates the decision tree on the given input. Must only be
        called after the tree has been built.
        :param input: Input point. Should not contain the last component
        which corresponds to the output label.
        :return: Output label predicted by the decision tree."""
        return self.root.evaluate(input)

    def __str__(self):
        return self.root.description_string(0)


class Node:
    """A node of the decision tree."""

    def __init__(self, attr_index=None):
        self.successors = []
        self.attr_index = attr_index

    @staticmethod
    def suitable_node(data_points, depth, depth_constraint, gain_threshold):
        """Constructs a suitable node for the given data points. If a
        further test results in sufficient information gain and does not
        exceed the depth constraint, an inner node with the corresponding
        test (continuous or class attribute) is created. Otherwise, a
        Leaf with a suitable output label is created.
        :param data_points: List of tuples representing the data points.
        Last tuple component must corresponds to output label.
        :param depth: Depth of the node to be constructed.
        :param depth_constraint: Specifies the maximum depth allowed
        for the decision tree.
        :param gain_threshold: Minimum information gain a test/ inner node
        must achieve to be added to the tree. Otherwise, no further test
        is added to the current branch.
        :return: Suitable node."""
        output_class_sizes = partition_sizes_class_attribute(data_points, -1)
        output_classes = len(output_class_sizes)
        if output_classes == 1:   # Recursion ends
            return Leaf(data_points[0][-1])
        else:                     # Recursion continues (with exception)
            best_split_attr, gain = best_split_attribute(data_points)
            if gain > gain_threshold and (depth_constraint is None or
                                          depth <= depth_constraint):
                if type(data_points[0][best_split_attr]) is int:
                    return ClassNode(data_points, best_split_attr, depth,
                                     depth_constraint, gain_threshold)
                else:
                    return ContNode(data_points, best_split_attr, depth,
                                    depth_constraint, gain_threshold)
            else:                 # Gain too small, Recursion ends
                value = max(output_class_sizes.items(),
                            key=lambda item: item[1])[0]
                return Leaf(value)


class ContNode(Node):
    """A node of the decision tree which performs a test on a continuous
    attribute."""

    def __init__(self, data_points, attr_index, depth, depth_constraint,
                 gain_threshold):
        """Initializes a node for a continuous attribute test using the
        given data points.
        :param data_points: List of tuples representing the data points.
        Last tuple component must corresponds to output label.
        :param attr_index: Index of the attribute used for the test.
        Must be a continuous attribute.
        :param depth: Depth of the node to be constructed.
        :param depth_constraint: Specifies the maximum depth allowed
        for the decision tree.
        :param gain_threshold: Minimum information gain a test/ inner node
        must achieve to be added to the tree. Otherwise, no further test
        is added to the current branch."""
        Node.__init__(self, attr_index)
        self.split_value = cont_attr_best_split(data_points, attr_index)[1]
        partitioned_data_points = split(data_points, attr_index,
                                        self.split_value)
        self.__create_successors(partitioned_data_points, depth + 1,
                                 depth_constraint, gain_threshold)

    def __create_successors(self, partitioned_data_points, depth,
                            depth_constraint, gain_threshold):
        """Adds suitable successor nodes to the current node.
        :param partitioned_data_points: Dictionary of partitions where each
        partition contains a list of tuples representing the data points.
        :param depth: Depth of the node to be constructed.
        :param depth_constraint: Specifies the maximum depth allowed
        for the decision tree.
        :param gain_threshold: Minimum information gain a test/ inner node
        must achieve to be added to the tree. Otherwise, no further test
        is added to the current branch."""
        for partition in partitioned_data_points:
            successor = Node.suitable_node(partition, depth, depth_constraint,
                                           gain_threshold)
            self.successors.append(successor)

    def evaluate(self, input):
        """Evaluates the subtree rooted in the current node on the given
        input by performing the test on the input and passes the input
        on to its successors.
        :param input: Input point. Should not contain the last component
        which corresponds to the output label.
        :return: Output label predicted by this subtree."""
        if input[self.attr_index] <= self.split_value:
            return self.successors[0].evaluate(input)
        else:
            return self.successors[1].evaluate(input)

    def description_string(self, depth):
        """String representation of the subtree rooted in the current node.
        :param depth: Depth of the current node.
        :return: String representation."""
        descr = ""
        indent = "\t" * depth
        descr += indent + "If x[" + str(self.attr_index) + "] <= " + \
                 str(self.split_value) + ":\n"
        descr += self.successors[0].description_string(depth + 1)
        descr += indent + "If x[" + str(self.attr_index) + "] > " + \
                 str(self.split_value) + ":\n"
        descr += self.successors[1].description_string(depth + 1)
        return descr


class ClassNode(Node):
    """A node of the decision tree which performs a test on a class
    attribute."""

    def __init__(self, data_points, attr_index, depth, depth_constraint,
                 gain_threshold):
        """Initializes a node for a class attribute test using the
        given data points.
        :param data_points: List of tuples representing the data points.
        Last tuple component must corresponds to output label.
        :param attr_index: Index of the attribute used for the test. Must
        be a class attribute.
        :param depth: Depth of the node to be constructed.
        :param depth_constraint: Specifies the maximum depth allowed
        for the decision tree.
        :param gain_threshold: Minimum information gain a test/ inner node
        must achieve to be added to the tree. Otherwise, no further test
        is added to the current branch."""
        Node.__init__(self, attr_index)
        partitioned_data_points = partitions_class_attribute(data_points,
                                                             attr_index)
        self.keys = list(partitioned_data_points.keys())
        self.__create_successors(partitioned_data_points, depth + 1,
                                 depth_constraint, gain_threshold)

    def __create_successors(self, partitioned_data_points, depth,
                            depth_constraint, gain_threshold):
        """Adds suitable successor nodes to the current node.
        :param partitioned_data_points: Dictionary of partitions where each
        partition contains a list of tuples representing the data points.
        :param depth: Depth of the node to be constructed.
        :param depth_constraint: Specifies the maximum depth allowed
        for the decision tree.
        :param gain_threshold: Minimum information gain a test/ inner node
        must achieve to be added to the tree. Otherwise, no further test
        is added to the current branch."""
        for key in self.keys:
            partition = partitioned_data_points[key]
            successor = Node.suitable_node(partition, depth, depth_constraint,
                                           gain_threshold)
            self.successors.append(successor)

    def evaluate(self, input):
        """Evaluates the subtree rooted in the current node on the given
        input by performing the test on the input and passes the input
        on to its successors.
        :param input: Input point. Should not contain the last component
        which corresponds to the output label.
        :return: Output label predicted by this subtree."""
        succ_index = 0
        if input[self.attr_index] in self.keys:
            succ_index = self.keys.index(input[self.attr_index])
        return self.successors[succ_index].evaluate(input)

    def description_string(self, depth):
        """String representation of the subtree rooted in the current node.
        :param depth: Depth of the current node.
        :return: String representation."""
        descr = ""
        indent = "\t" * depth
        for index, key in enumerate(self.keys):
            descr += indent + "If x[" + str(self.attr_index) + "] == " + \
                     str(key) + ":\n"
            descr += self.successors[index].description_string(depth + 1)
        return descr


class Leaf(Node):
    """Leaf of the decision tree. Holds a value for the output label which
    is returned when an input's evaluation leads to this leaf."""

    def __init__(self, value):
        Node.__init__(self)
        self.value = value

    def evaluate(self, input):
        """Evaluates the subtree rooted in the current node on the given
        input by performing the test on the input and passes the input
        on to its successors.
        :param input: Input point. Should not contain the last component
        which corresponds to the output label.
        :return: Output label predicted by this subtree."""
        return self.value

    def description_string(self, depth):
        """String representation of the subtree rooted in the current node.
        :param depth: Depth of the current node.
        :return: String representation."""
        return "\t" * depth + str(self.value) + "\n"


if __name__ == '__main__':
    tree = ClassificationTree()
    data = [(0, 1, 0, 0.25, 0), (0, 1, 1, 0.2, 0), (0, 2, 0, 0.5, 1),
            (0, 1, 0, 0.55, 1), (0, 2, 0, 0.52, 1), (0, 1, 1, 0.88, 2),
            (1, 1, 0, 0.95, 2)]
    tree.build(data)
    print(tree)
    print(tree.evaluate((0, 1, 1, 0.9)))
