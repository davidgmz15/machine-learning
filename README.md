# p4-machinelearning

## Overview
This C++ program uses natural language processing (NLP) and machine learning techniques to automatically identify the subject of EECS 280 Piazza posts. It predicts the topic (e.g., exam, euchre, recursion) of a given post by analyzing the words it contains and comparing them to a trained dataset.

## Features
- Implements a Bag-of-Words model with a Naive Bayes classifier.
- Supports training on labeled Piazza posts and predicting topics for new posts.
- Uses `csvstream.hpp` to read CSV files for both training and testing data.
- Provides detailed output of training statistics and prediction performance.

## Usage
```bash
# Compile
make classifier.exe

# Train only
./classifier.exe train.csv

# Train and test
./classifier.exe train.csv test.csv
