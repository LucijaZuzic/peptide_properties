#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
DATA_PATH = '../data/'
MODEL_DATA_PATH = {
    'AP_ALL_PROPERTIES': '../model_data/non_merge/AP_ALL_PROPERTIES/', 
    'ALL_PROPERTIES': '../model_data/non_merge/ALL_PROPERTIES/', 
    'AP': '../model_data/non_merge/AP/', 
    'APH': '../model_data/non_merge/APH/', 
    'logP': '../model_data/non_merge/logP/',
    'polarity_relu': '../model_data/non_merge/polarity_relu/', 
    'polarity_selu': '../model_data/non_merge/polarity_selu/'
    } 

MERGE_MODEL_DATA_PATH = {
    'AP_logP': '../model_data/merge/AP_logP/', 
    'AP_polarity_relu': '../model_data/merge/AP_polarity_relu/', 
    'AP_polarity_selu': '../model_data/merge/AP_polarity_selu/', 
    'APH_logP': '../model_data/merge/APH_logP/', 
    'APH_polarity_relu': '../model_data/merge/APH_polarity_relu/', 
    'APH_polarity_selu': '../model_data/merge/APH_polarity_selu/', 
    }

def scale(AP_dictionary, offset = 0.5):
    data = [AP_dictionary[key] for key in AP_dictionary]

    # Determine min and max AP scores.
    min_val = min(data)
    max_val = max(data)

    # Scale AP scores to range [- offset, 1 - offset].
    for key in AP_dictionary:
        AP_dictionary[key] = (AP_dictionary[key] - min_val) / (max_val - min_val) - offset
   
def split_amino_acids(sequence, amino_acids_AP_scores):
    ap_list = []

    # Replace each amino acid in the sequence with a corresponding AP score.
    for letter in sequence:
        ap_list.append(amino_acids_AP_scores[letter])

    return ap_list

def split_dipeptides(sequence, dipeptides_AP_scores):
    ap_list = []

    # Replace each dipeptide in the sequence with a corresponding AP score.
    for i in range(len(sequence) - 1):
        ap_list.append(dipeptides_AP_scores[sequence[i:i + 2]])

    return ap_list

def split_tripeptides(sequence, tripeptides_AP_scores):
    ap_list = []

    # Replace each tripeptide in the sequence with a corresponding AP score.
    for i in range(len(sequence) - 2):
        ap_list.append(tripeptides_AP_scores[sequence[i:i + 3]])

    return ap_list

def pick_data_kfold(data, test_p):
    # Calculate the number of instances for test dataset.
    n_test = int(test_p * len(data))

    # Separate the test subset from the train and validation subsets.
    test = data[:n_test]
    kfold_data = data[n_test:]

    return test, kfold_data

def split_data_kfold(SA, NSA, test_p):
    # Randomly shuffle the data in place.
    np.random.shuffle(SA)
    np.random.shuffle(NSA)
    
    # Separate the self assembly test subset from the self assembly train subset and self assembly validation subset.
    test_SA, kfold_data_SA = pick_data_kfold(SA, test_p)

    # Separate the non self assembly test subset from the non self assembly train subset and non self assembly validation subset.
    test_NSA, kfold_data_NSA  = pick_data_kfold(NSA, test_p)

    # Make test subset.
    test_data = test_SA + test_NSA
    test_labels = np.ones(len(test_SA) + len(test_NSA))
    test_labels[len(test_SA):] *= 0

    # Make the train and validation subsets.
    kfold_data = kfold_data_SA + kfold_data_NSA
    kfold_labels = np.ones(len(kfold_data_SA) + len(kfold_data_NSA))
    kfold_labels[len(kfold_data_SA):] *= 0

    return test_data, test_labels, kfold_data, kfold_labels

