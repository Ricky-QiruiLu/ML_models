"""
Created on Sat Sep 16 2021

@author: Qirui Lu (Ricky)
"""

import sys
import math


def retrive_rows(file_name):
    """
    get the attributes and rows of data from the tsv file
    
    :param file_name: training or testing data file

    :return: a list of attributes, and a list of rows(list structure)

    """
    attrs = []
    data = []
    file = open(file_name)
    for i, line in enumerate(file):
        sample = line.split()
        if (i == 0):
            attrs = sample
        else:
            data.append(sample)

    return attrs, data


def count_on_attr(dataset, attr):
    """
    count the labels for attribute attr(an index)
    
    :param dataset: a list of row data

    :param attr: the index of the attribute

    :return: (a dictionary with counts, a list of keys for the dic)

    """
    result = {}
    for sample in dataset:
        if (sample[attr] not in result):
            result[sample[attr]] = 0
        result[sample[attr]] += 1

    return (result, list(result.keys()))


def get_subset(dataset, attr, label):
    """
    get a subset from the dataset which attribute attr is label

    :param dataset: a list of row data

    :param attr: the index of the attribute

    :label: the desired label for the attribute attr

    :return: a list of row data

    """
    return [item for item in dataset if item[attr] == label]


def get_split_result(dataset, attr):
    """
    get the result after split with attribute attr

    :param dataset: a list of row data

    :param attr: the index of the attribute to split

    :return: a dictionary with split result, i.e. {n: {'democrat': 13, 'republican': 58}, y...}

    """
    result = {}
    for sample in dataset:
        if sample[attr] not in result:
            result[sample[attr]] = {}

        if sample[-1] not in result[sample[attr]]:
            result[sample[attr]][sample[-1]] = 0
        result[sample[attr]][sample[-1]] += 1
    return result


def get_print_data(data, keys):
    """
    get the printed data

    :param data: a dictionary with data for the current node

    :param keys: the labels that the current splited node has

    :return: a string i.e. [1 democrat/1 republican]

    """
    result = "["
    for key in keys:
        if key not in data:
            result += '0 ' + key + '/'
        else:
            result += str(data[key]) + ' ' + key + '/'

    result = result[:-1] + ']'
    return result


def print_node(attr, keys, label, data, depth):
    """
    print the information for each Node

    :param attr: the splited attribute

    :param keys: the labels that the current splited node has

    :param label: the label for the node

    :param data: the information for the node 

    :param depth: current depth
    """
    result = ""
    count = 0
    while count <= depth:
        result += "| "
        count += 1
    result += attr + ' = ' + label + ": "

    result += get_print_data(data, keys)
    print(result)


class Node(object):
    def __init__(self, depth, max_depth, dataset, remain_attrs):

        self.remain_attrs = remain_attrs  # available attributes to be splitted
        self.max_depth = max_depth
        self.depth = depth
        self.dataset = dataset
        self.entropy = self.get_entropy(dataset)
        self.split_attr = self.find_split_attr(
        )  # -1 for no splitted attribute
        if (self.split_attr != -1):

            self.keys = count_on_attr(
                self.dataset,
                self.split_attr)[1]  #get the labels for the split_attr
            self.keys.sort()

            self.remain_attrs.remove(
                self.split_attr
            )  #the split_attr will not be available for children
            children_data = get_split_result(self.dataset,
                                             self.split_attr)  #split the data
            leaf_data = self.majority_split()  #get the result

            # if we reach the max depth or no attributes available, then we construct Leaf for children
            if (depth == self.max_depth - 1 or len(remain_attrs) == 0):
                print_node(attrs_named[self.split_attr], head_labels,
                           self.keys[0], children_data[self.keys[0]],
                           self.depth)
                print_node(attrs_named[self.split_attr], head_labels,
                           self.keys[1], children_data[self.keys[1]],
                           self.depth)
                self.left_node = Leaf(leaf_data[self.keys[0]])
                self.right_node = Leaf(leaf_data[self.keys[1]])

            else:
                self.left_node = self.generate_child(children_data,
                                                     self.remain_attrs.copy(),
                                                     0)
                if isinstance(self.left_node,
                              Node) and self.left_node.split_attr == -1:
                    self.right_node = Leaf(
                        leaf_data[self.keys[0]]
                    )  # for split_attr = -1, we just want it to be Leaf
                self.right_node = self.generate_child(children_data,
                                                      self.remain_attrs.copy(),
                                                      1)
                if isinstance(self.right_node,
                              Node) and self.right_node.split_attr == -1:
                    self.right_node = Leaf(leaf_data[self.keys[1]])

    def get_entropy(self, dataset):
        """
        get the entropy for the dataset

        :param dataset: the data set

        :return: the entropy
        """
        data = count_on_attr(dataset, -1)[0]
        if len(data) == 1:
            return 0
        total_number = sum(list(data.values()))
        result = 0.0
        for label in data:
            result -= math.log2(
                data[label] / total_number) * (data[label] / total_number)
        return result

    def get_condi_entropy(self, X):
        """
        get the conditional entropy for  self.dataset

        :param X: the attribute for condition

        :return: the conditional entropy
        """
        ratio = count_on_attr(self.dataset, X)[0]
        result = 0.0

        for label in ratio:
            subset = get_subset(self.dataset, X, label)
            result += (ratio[label] /
                       len(self.dataset)) * self.get_entropy(subset)
        return result

    def get_mutual_information(self, X):
        """
        get the mutual information for  self.dataset

        :param X: the attribute for condition

        :return: the mutual information
        """
        condi_entropy = self.get_condi_entropy(X)
        return self.entropy - condi_entropy

    def find_split_attr(self):
        """
        find the index of next splitted attribute, 
        return -1 if all attributes have MI of 0
        """
        max_MI = 0.0
        result = self.remain_attrs[0]

        for attr in self.remain_attrs:
            temp_MI = self.get_mutual_information(attr)
            if temp_MI > max_MI and temp_MI > 0:
                max_MI = temp_MI
                result = attr
        if max_MI == 0.0:
            return -1
        return result

    # def majority_split(self):
    #     """
    #     do a majority split with splitted_attr

    #     :return: a dictionary with result, i.e. {'y': 'democrat', 'n': 'republican'}
    #     """
    #     freq_dic = get_split_result(self.dataset, self.split_attr)
    #     result = {}
    #     for attr in freq_dic:
    #         sorted_dic = dict(
    #             sorted(freq_dic[attr].items(),
    #                    key=operator.itemgetter(1),
    #                    reverse=True))
    #         result[attr] = list(sorted_dic.keys())[0]

    #     return result

    def majority_split(self):
        """
        do a majority split with splitted_attr
        when there is a tie in the number of occurance, chose the label with larger value
        i.e. when A and notA are both 10 times occurance, choose not A

        :return: a dictionary with result, i.e. {'y': 'democrat', 'n': 'republican'}
        """
        freq_dic = get_split_result(self.dataset, self.split_attr)
        result = {}
        for attr in freq_dic:
            keys = list(freq_dic[attr].keys())
            keys.sort()
            max_val = 0
            for val in keys:
                if max_val <= freq_dic[attr][val]:
                    max_val = freq_dic[attr][val]
                    result[attr] = val
        return result

    def generate_child(self, children_data, remain_attrs, idx):
        """
        find the child, either a Leaf or a Node

        :children_data: the split result with splitting on the split_attr

        :remain_attrs: the available attributes to be splited in the child

        :idx: which label of attribute, 0 or 1

        :return: next child
        """
        data = children_data[self.keys[idx]]
        print_node(attrs_named[self.split_attr], head_labels, self.keys[idx],
                   children_data[self.keys[idx]], self.depth)

        if len(data) == 1:
            label = list(data.keys())[0]
            return Leaf(label)

        subset = get_subset(self.dataset, self.split_attr, self.keys[idx])
        return Node(self.depth + 1, self.max_depth, subset, remain_attrs)


class Leaf(object):
    def __init__(self, label):
        self.label = label


def make_prediction(sample, root):
    """
    predict the sample result with decision tree

    :param sample: a row of data

    :root: the root node of the decisiontree

    :return: the predicted label
    """
    curr = root
    while (isinstance(curr, Node)):
        curr_label = sample[curr.split_attr]
        if (len(curr.keys) < 2):
            curr = curr.left_node
            break

        if (curr_label == curr.keys[0]):
            attrs_named
            curr = curr.left_node
        else:
            curr = curr.right_node

    return curr.label


def get_metrics(all_data, root):
    """
    get the metric data (error rate) for all_data

    :param all_data: the data

    :root: the root node of the decisiontree

    :return: a string of error rate
    """
    count = 0.0
    errors = 0
    for sample in all_data:
        count += 1
        predict = make_prediction(sample, root)
        if (predict != sample[-1]):
            errors += 1
    return str(errors / count)


def generate_predic_file(output_file, data, root):
    f = open(output_file, "w")
    f.write(make_prediction(data[0], root))

    for sample in data[1:]:
        prediction = make_prediction(sample, root)
        f.write('\n' + prediction)
    f.close()


def generate_matrics_file(output_file, data_train, data_test, root):
    f = open(output_file, "w")
    line1 = "error(train): " + get_metrics(data_train, root) + '\n'
    line2 = "error(test): " + get_metrics(data_test, root)

    print(line1, line2)
    f.write(line1)
    f.write(line2)
    f.close()


if __name__ == '__main__':

    # pass the inputs
    training_file = sys.argv[1]
    test_file = sys.argv[2]
    MAX_DEPTH = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    # get the training data, attribute names, attribute indexes, test data
    attrs_named, train_data = retrive_rows(training_file)
    attrs_idx = list(range(len(attrs_named) - 1))
    test_data = retrive_rows(test_file)[1]

    # print the first line of the decision tree
    head_data, head_labels = count_on_attr(train_data, -1)
    print(get_print_data(head_data, list(head_data.keys())))

    # construct the tree
    root = Node(0, MAX_DEPTH, train_data, attrs_idx)

    # generate output files
    generate_predic_file(train_out, train_data, root)
    generate_predic_file(test_out, test_data, root)
    generate_matrics_file(metrics_out, train_data, test_data, root)