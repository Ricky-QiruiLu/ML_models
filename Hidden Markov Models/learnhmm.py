"""
Created on Wednesday Nov 11 2021

@author: Qirui Lu (Ricky)
"""

import numpy as np
import sys

def construct_indices(file):
    """
    convert the index_words and index_tags into dictionary
    
    :param file: data file

    :return: {word1: 0, word2: 1 ...}

    """
    result = {}
    with open(file) as f:
        index = 0
        for line in f:
            result[line.strip()] = index
            index += 1
    
    return result

def convert_training_data(file):
    """
    convert the training sequences into lists
    sequences are in format of [(word1, tag1), (word2, tag2)]
    
    :param file: data file

    :return: [[seq1]. [seq2]]

    """
    result = []
    temp_string = []
    with open(file) as f:
        for line in f:
            if line == "\n":
                result.append(temp_string)
                temp_string = []
                continue

            data = line.strip().split("\t")
            temp_string.append((data[0], data[1]))
        result.append(temp_string)  # add the last string
    
    return result

def construct_hmminit(training_data, tags):
    """
    convert the training data to hmm init list
    
    :param training_data: the training data

    :param tags: the tags 

    :return: ndarray [p1, p2, ...]

    """
    result = np.ones(len(tags))

    for seq in training_data:
        tag_idx = tags[seq[0][1]]
        result[tag_idx] += 1
    
    result = result / sum(result)
    return result

def write_hmminit(init_matrix, output_file):
    with open(output_file, "w") as f:
        for i in range(len(init_matrix)):
            f.write("{:.18e}".format(init_matrix[i]) + "\n")
    

def construct_hmmemit(training_data, tags, words):
    """
    convert the training data to hmm emit list
    
    :param training_data: the training data

    :param tags: the tags 

    :param words: the word list 

    :return: ndarray

    """
    result = np.ones((len(tags), len(words)))

    for seq in training_data:
        for word_tag in seq:
            idx_word = words[word_tag[0]]
            idx_tag = tags[word_tag[1]]
            result[idx_tag][idx_word] += 1
    
    for i in range(len(result)):
        result[i] = result[i] / sum(result[i])
    return result

def write_hmmemit_or_hmmtrans(matrix, output_file):
    with open(output_file, "w") as f:
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if j == len(matrix[i]) - 1:
                    f.write("{:.18e}".format(matrix[i][j]))
                    break
                f.write("{:.18e}".format(matrix[i][j]) + " ")
            f.write("\n")

def construct_hmmtrans(training_data, tags):
    """
    convert the training data to hmm trans list
    
    :param training_data: the training data

    :param tags: the tags 

    :return: ndarray

    """    
    result = np.ones((len(tags), len(tags)))

    for seq in training_data:
        for i in range(len(seq) - 1):
            idx_t = tags[seq[i][1]]
            idx_t_1 = tags[seq[i+1][1]]
            result[idx_t][idx_t_1] += 1
    
    for i in range(len(result)):
        result[i] = result[i] / sum(result[i])
    return result

def main():
    training_data = convert_training_data(train_input)
    tags = construct_indices(index_to_tag)
    words = construct_indices(index_to_word)

    init_matrix = construct_hmminit(training_data, tags)
    emit_matrix = construct_hmmemit(training_data, tags, words)
    trans_matrix = construct_hmmtrans(training_data, tags)

    write_hmminit(init_matrix, hmminit)
    write_hmmemit_or_hmmtrans(emit_matrix, hmmemit)
    write_hmmemit_or_hmmtrans(trans_matrix, hmmtrans)


if __name__ == '__main__':
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmminit = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    main()