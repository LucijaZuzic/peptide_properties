#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
DATA_PATH = '../data/'
MODEL_DATA_PATH = '../model_data/'
SEQ_MODEL_DATA_PATH = '../seq_model_data/'
MY_MODEL_DATA_PATH = '../only_my_model_data/'

def scale(AP_dictionary, offset = 1):
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

def padding(array, len_to_pad, value_to_pad):
    new_array = [value_to_pad for i in range(len_to_pad)]
    for val_index in range(len(array)):
        if val_index < len(new_array):
            new_array[val_index] = array[val_index]
    return new_array

def split_tripeptides(sequence, tripeptides_AP_scores):
    ap_list = []

    # Replace each tripeptide in the sequence with a corresponding AP score.
    for i in range(len(sequence) - 2):
        ap_list.append(tripeptides_AP_scores[sequence[i:i + 3]])

    return ap_list