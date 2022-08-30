"""
Created on Wednesday Nov 11 2021

@author: Qirui Lu (Ricky)
"""

import sys
import numpy as np

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

def convert_validation_input(file):
    """
    convert the validation sequences into lists
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

def construct_pi(hmminit):
    """
    construct the pi lists
    
    :param hmminit: data file

    :return: pi list

    """
    pi = []
    with open(hmminit) as f:
        for val in f:
            pi.append(float(val.strip()))
    return pi

def construct_A_or_B(matrix_file):
    """
    construct the A or B metrix
    
    :param matrix_file: data file

    :return: the ndarray matrix

    """
    result = []
    with open(matrix_file) as f:
        for row in f:
            data = row.strip().split(" ")
            data = [float(x) for x in data]
            result.append(data)
    
    result = np.array(result)


    return result

def log_sum_exp(alpha_t):
    """
    log(sum(expa + expb +...))
    in order to prevent underflow
    
    :param alpha_t: the data we want to sum

    :return: ndarray

    """
    n = len(alpha_t)
    m = np.amax(alpha_t)
    result = m + np.log(sum(np.exp(alpha_t - m*np.ones(n))))
    return result

def get_next_alpha(alpha_t, A, B, word_idx):
    """
    helper to get the next (t+1) alpha
    
    :param alpha_t: the current alpha t

    :param A: the emission matrix A

    :param B: the transition matrix B

    :param word_idx: the current X_t word index

    :return: ndarray

    """
    repeats_alpha = np.tile(alpha_t, (len(B), 1))
    repeats_alpha = repeats_alpha.T
    v = repeats_alpha + np.log(B)
    sigma = log_sum_exp(v)
    return np.log(A[:,word_idx]) + sigma

def generate_alpha(sequence, words, pi, A, B):
    idx_x1 = words[sequence[0][0]]
    log_alpha = []
    log_alpha.append(np.log(pi) + np.log(A[:, idx_x1]))  # aplha1
    
    for t in range(1, len(sequence)):
        word_idx = words[sequence[t][0]]
        alpha_t = log_alpha[t-1]
 
        new_alpha = get_next_alpha(alpha_t, A, B, word_idx)
        log_alpha.append(new_alpha)
    return log_alpha

def get_next_beta(beta_t, A, B, word_idx):
    repeated_beta_t = np.tile(beta_t, (len(B), 1))
    repeated_A_t = np.tile(A[:, word_idx], (len(B), 1))
    v = np.log(repeated_A_t) + repeated_beta_t + np.log(B)
    v = v.T
    sigma = log_sum_exp(v)
    return sigma

def generate_beta(sequence, words, pi, A, B):
    log_beta = []
    log_beta.append(np.zeros(len(B)))
    
    for t in range(len(sequence) - 1)[::-1]:
        word_idx = words[sequence[t + 1][0]]
        beta_t = log_beta[-1]

        new_beta = get_next_beta(beta_t, A, B, word_idx)
        log_beta.append(new_beta)

    log_beta.reverse()
    return log_beta

def get_mean_log_likelihood(pi, A, B, words):
    """
    get the mean log likelihood over all the validation data
    
    :param pi: the estimated initialization probabilities pi matrix

    :param A: the emission matrix A

    :param B: the transition matrix B

    :param words: the word dictionary

    :return: mean log likelihood

    """
    log_likelihood = 0.0
    for sequence in validation_data:
        alpha = generate_alpha(sequence, words, pi, A, B)
        log_likelihood += log_sum_exp(alpha[-1])
    
    log_likelihood = log_likelihood / len(validation_data)
    return log_likelihood

def get_tag_list(index_to_tag):
    result = []
    with open(index_to_tag) as f:
        index = 0
        for tag in f:
            result.append(tag.strip())
    return result

def predict(sequence, tag_list, alpha, beta):
    """
    the simple method to obtain the predicted tag for 
    X_t by calculating alpha_t * beta_t and get the 
    maximum unit as the corresponding tag

    :param sequence: the sequence to predict

    :param tag_list: the tags

    :param alpha: the alpha matrix

    :param beta: the beta matrix

    :return: the predicted tag list

    """
    result = []
    for t in range(len(sequence)):
        temp = alpha[t] + beta[t]
        max_idx = np.where(temp == np.amax(temp))[0][0] # get the first max value index
        result.append(tag_list[max_idx])
    return result


def get_predicted_tag_seq(b, max_idx, tag_list):
    """
    predict the tags according to matrix b
    
    :param max_idx: the index we chooose for b[-1]

    :param tag_list: the tags

    :return: the prediction

    """
    result = []
    result.append(tag_list[max_idx])    # the last tag

    temp_tag = b[-1][max_idx]
    for i in range(1,len(b))[::-1]:
        result.append(temp_tag)
        idx = tag_list.index(temp_tag)
        temp_tag = b[i-1][idx]
    result.append(temp_tag)

    result = result[::-1]
    return result

def viterbi_predict_helper(sequence, w, b, tag_list, t):
    if t == len(sequence):  # base case, where we need to return the prediction
        last_t = w[-1]
        max_w = 0.0
        max_idx = 0
        for i in range(len(last_t)):
            if max_w < last_t[i]:
                max_idx = i
                max_w = last_t[i]   # find the maximal index in the last row of w
        return get_predicted_tag_seq(b, max_idx, tag_list)
    

    elif t == 0:    # base case, where we need to create the first row in w
        curr_word = sequence[0][0]
        for tag in tag_list:
            idx_word = words[curr_word]
            idx_tag = tags[tag]
            w[0] = A[:, idx_word] * pi  # construct w[0]
        
    else:
        curr_word = sequence[t][0]
        b_list = []
        for tag in tag_list:
            idx_word = words[curr_word]
            idx_tag = tags[tag]
            temp = A[idx_tag, idx_word] * w[t - 1] * B[:, idx_tag]
            w[t][idx_tag] = np.amax(temp)   # get the best choice for w[t][idx_tag]
            max_idx = np.where(temp == w[t][idx_tag])[0][0]
            b_list.append(tag_list[max_idx])    # record the selection in b[t-1]
        
        b.append(b_list)
    
    return viterbi_predict_helper(sequence, w, b, tag_list, t+1)

def viterbi_predict(sequence, tag_list):
    """
    this is the viterbi prediction for the sequence

    :param sequence: the sequence to predict

    :param tag_list: the tags

    :return: the predicted tag list

    """
    w = np.zeros((len(sequence), len(tag_list)))
    b = []
    return viterbi_predict_helper(sequence, w, b, tag_list, 0)



def get_true_and_predict_sequences(validation_data, tag_list):
    """
    get the true and predicted tags
    
    :param validation_data: the data

    :param tag_list: the tags list
    
    :return: a true list and a prediction list

    """
    true = []
    prediction = []

    for seq in validation_data:
        temp = []
        for val in seq:
            temp.append(val[1])
        true.append(temp)   # add true tags

        alpha = generate_alpha(seq, words, pi, A, B)
        beta = generate_beta(seq, words, pi, A, B)
        one_prediction = predict(seq, tag_list, alpha, beta)

        prediction.append(one_prediction)   # add prediction tags
    return true, prediction

def get_prediction_accuracy(validation_data, tag_list):
    correct = 0
    count = 0
    true, prediction = get_true_and_predict_sequences(validation_data, tag_list)

    for seq_t, seq_p in zip(true, prediction):
        for val_t, val_p in zip(seq_t, seq_p):
            if val_t == val_p:
                correct += 1
            count += 1
    return correct / count 

def write_prediction(validation_data, tag_list, predicted_file):
    true, prediction = get_true_and_predict_sequences(validation_data, tag_list)
    with open(predicted_file, "w") as f:
        for seq_t, seq_p in zip(validation_data, prediction):
            for val_t, val_p in zip(seq_t, seq_p):
                
                f.write(val_t[0] + "\t" + val_p + "\n")
            f.write("\n")

def write_metrics(validation_data, tag_list, metric_file):
    with open(metric_file, "w") as f:
        mean_log_likelihood = get_mean_log_likelihood(pi, A, B, words)
        accuracy = get_prediction_accuracy(validation_data, tag_list)
        f.write("Average Log-Likelihood: " + str(mean_log_likelihood) + "\n")
        f.write("Accuracy: " + str(accuracy))

def main():
    write_prediction(validation_data, tag_list, predicted_file)
    write_metrics(validation_data, tag_list, metric_file)

if __name__ == '__main__':

    validation_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmminit = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]

    # validation_input = "validation.txt"
    # index_to_word = "index_to_word.txt"
    # index_to_tag = "index_to_tag.txt"
    # hmminit = "hmminit.txt"
    # hmmemit = "hmmemit.txt"
    # hmmtrans = "hmmtrans.txt"
    # predicted_file = "my_predict.txt"
    # metric_file = "my_metric.txt"

    # the variables
    validation_data = convert_validation_input(validation_input)
    tag_list = get_tag_list(index_to_tag)
    tags = construct_indices(index_to_tag)
    words = construct_indices(index_to_word)

    pi = construct_pi(hmminit)
    A = construct_A_or_B(hmmemit)
    B = construct_A_or_B(hmmtrans)

    main()

 