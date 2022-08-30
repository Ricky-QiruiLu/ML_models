# Hidden Markov Models

## Overview

In this programming section, I implement a named entity recognition system using Hidden Markov
Models (HMMs).

## Inputs

There are 3 provided files:

1. validation.txt, this file contains labeled data that I use to evaluate your model in the Experiments section.
2. index_to_word.txt and index_to_tag.txt, these files contain a list of all words or tags that appear in the data set.

In order to learn the HMM model, we need to first generate 3 files: hmminit.txt (the initialization probabilities), hmmtrans.txt (the transition probabilities), and hmmemit.txt (the emission probabilities)

To generate the above 3 files, we can run

```
python3 learnhmm.py [args...]
```

args are: train_input, index_to_word, index_to_tag, hmminit, hmmemit, hmmtrans

1. train_input: path to the training input .txt file
2. index_to_word: path to the .txt that specifies the dictionary mapping from words to indices. The tags are ordered by index, with the first word having index of 0, the second word having index of 1, etc.
3. index_to_tag: path to the .txt that specifies the dictionary mapping from tags to indices. The tags are ordered by index, with the first tag having index of 0, the second tag having index of 1, etc.
4. hmminit: path to output .txt file to which the estimated initialization probabilities (π) will be written. The file output to this path should be in the same format as the handout hmminit.txt.
5. hmmemit: path to output .txt file to which the emission probabilities (A) will be written. The file output to this path should be in the same format as the handout hmmemit.txt.
6. hmmtrans: path to output .txt file to which the transition probabilities (B) will be written. The file output to this path should be in the same format as the handout hmmtrans.txt.

Then, we can simply run the forwardbackward algorithm:

```
python3 forwardbackward.py [args...]
```

[args...] is a placeholder for seven command-line arguments: validation_input, index_to_word, index_to_tag, hmminit, hmmemit, hmmtrans, predicted_file, metric_file.

## outputs

the output files for the prediction and the metrics
