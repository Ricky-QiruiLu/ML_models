# Decision Tree


## Overview

This file learns a decision
tree with a specified maximum depth, print the decision tree in a specified format, predict the labels of the
training and testing examples, and calculate training and testing errors.

## Inputs

```
python3 decisionTree.py [args...]
```

args are: train_input, test_input, max_depth, train_out, test_out, metrics_out

1. train_input: path to the training input .tsv file 
2. test_input: path to the test input .tsv file 
3. max_depth: maximum depth to which the tree should be built
4. train_out: path of output .labels file to which the predictions on the training data should
be written 
5. test_out: path of output .labels file to which the predictions on the test data should be
written
6. metrics_out: path of the output .txt file to which metrics such as train and test error should
be written

for example, the command line could be: 
```
python3 decisionTree.py small_train.tsv small_test.tsv 2 small_2_train.labels small_2_test.labels small_2_metrics.txt
```

## outputs
the terminal will print out the tree, here is an example:

![tree_sample](decision_tree\tree_example.png)


the label files (train_out, test_out) and the metric file


