import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import PATH_TO_EXTENSION, DATA_PATH, SEQ_MODEL_DATA_PATH, MODEL_DATA_PATH, MY_MODEL_DATA_PATH, setSeed, predictions_name
from custom_plots import merge_type_iteration 
from scipy import stats
import sklearn 
from utils import predictions_name, final_history_name, DATA_PATH, SEQ_MODEL_DATA_PATH, MODEL_DATA_PATH, MY_MODEL_DATA_PATH, setSeed, getSeed, PATH_TO_EXTENSION
from custom_plots import merge_type_iteration, results_name,my_accuracy_calculate, weird_division, convert_to_binary
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    auc,
    f1_score,
)

def read_ROC(test_labels, model_predictions, lines_dict): 
    # Get false positive rate and true positive rate.
    fpr, tpr, thresholds = roc_curve(test_labels, model_predictions) 

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ix = np.argmax(gmeans) 

    lines_dict['ROC thr = '].append(thresholds[ix])
    lines_dict['gmean = '].append(gmeans[ix])
    lines_dict['ROC AUC = '].append(roc_auc_score(test_labels, model_predictions))
    lines_dict['Accuracy (ROC thr) = '].append(my_accuracy_calculate(test_labels, model_predictions, thresholds[ix]))
   
def read_PR(test_labels, model_predictions, lines_dict):  
    # Get recall and precision.
    precision, recall, thresholds = precision_recall_curve(
        test_labels, model_predictions
    ) 

    # Calculate the F1 score for each threshold
    fscore = []
    for i in range(len(precision)):
        fscore.append(
            weird_division(2 * precision[i] * recall[i], precision[i] + recall[i])
        )

    # Locate the index of the largest F1 score
    ix = np.argmax(fscore)
    model_predictions_binary_thr = convert_to_binary(model_predictions, thresholds[ix])
    model_predictions_binary = convert_to_binary(model_predictions, 0.5)
 
    lines_dict['PR thr = '].append(thresholds[ix])
    lines_dict['PR AUC = '].append(auc(recall, precision))
    lines_dict['F1 = '].append(fscore[ix])
    lines_dict['F1 (0.5) = '].append(f1_score(test_labels, model_predictions_binary))
    lines_dict['F1 (thr) = '].append(f1_score(test_labels, model_predictions_binary_thr))
    lines_dict['Accuracy (PR thr) = '].append(my_accuracy_calculate(test_labels, model_predictions, thresholds[ix]))
    lines_dict['Accuracy (0.5) = '].append(my_accuracy_calculate(test_labels, model_predictions, 0.5))

def arrayToTable(array, addArray, addAvg, addMostFrequent):
    retstr = ""
    suma = 0
    freqdict = dict()
    for i in array:
        if addArray:
            if addMostFrequent:
                retstr += " & %d" %(i)
            if addAvg:
                retstr += " & %.5f" %(i)
        suma += i
        if i in freqdict:
            freqdict[i] = freqdict[i] + 1
        else:
            freqdict[i] = 1
    most_freq = []
    num_most = 0
    for i in freqdict:
        if freqdict[i] >= num_most:
            if freqdict[i] > num_most:
                most_freq = []
            most_freq.append(i)
            num_most = freqdict[i] 
    if addAvg:
        retstr += " & %.5f" %(np.mean(array))
    if addMostFrequent: 
        most_freq_str = str(sorted(most_freq)[0])
        for i in sorted(most_freq[1:]):
            most_freq_str += ", " + str(i)
        retstr += " & %s (%d/%d)" %(most_freq_str, num_most, len(array)) 
    return retstr + " \\\\"

def hex_predictions_name(some_path, number, final_model_type, iteration):
    return predictions_name(some_path, number, final_model_type, iteration).replace("predictions", "hex_predictions")
 
def hex_predictions_png_name(some_path, number, final_model_type, iteration):
    return hex_predictions_name(some_path, number, final_model_type, iteration).replace(".txt", ".png")

def hex_predictions_final_name(some_path):
    return "../seeds/all_seeds/" + PATH_TO_EXTENSION[some_path] + "_all_tests_hex_predictions.png"

def hex_predictions_seed_name(some_path):
    return "../seeds/seed_" + str(seed) + some_path.replace("..", "") + PATH_TO_EXTENSION[some_path] + "_seed_" + str(seed) + "_hex_predictions.png"

df = pd.read_csv(DATA_PATH + "41557_2022_1055_MOESM3_ESM_Figure3a_5mer_score_shortMD.csv")

dict_hex = {}
for i in df['pep']:
    dict_hex[i] = '1' 

seq_example = ''
for i in range(24):
    seq_example += 'A'
dict_hex[seq_example] = '1' 

best_batch_size = 600
best_model = '' 
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276] 
NUM_TESTS = 5

offset = 1 
  
properties = np.ones(95)
masking_value = 2

actual_AP = []
actual_AP_long = []
test_labels = []
test_labels_long = []
threshold = 2
 
for i in df['AP']:
    actual_AP.append(i) 
    if i < threshold:
        test_labels.append(0) 
    else:
        test_labels.append(1) 

for number in range(1, NUM_TESTS + 1): 
    for i in df['AP']:
        actual_AP_long.append(i) 
        if i < threshold:
            test_labels_long.append(0) 
        else:
            test_labels_long.append(1) 
 
if not os.path.exists("../seeds/all_seeds/"):
    os.makedirs("../seeds/all_seeds/")

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
paths = [SEQ_MODEL_DATA_PATH, MODEL_DATA_PATH, MY_MODEL_DATA_PATH]
NUM_TESTS = 5

vals_in_lines = [ 
'ROC thr = ','ROC AUC = ', 'gmean = ', 
'PR thr = ', 'PR AUC = ', 'F1 = ', 'F1 (0.5) = ', 'F1 (thr) = ',
'Accuracy (0.5) = ', 'Accuracy (ROC thr) = ', 'Accuracy (PR thr) = '] 
  
for some_path in paths:

    lines_dict = dict()
    sd_dict = dict()

    for val in vals_in_lines:
        lines_dict[val] = [] 
        sd_dict[val] = []   

    for seed in seed_list:
        setSeed(seed)

        for number in range(1, NUM_TESTS + 1): 
            model_predictions_hex_file = open(hex_predictions_name(some_path, number, 'weak', 1), 'r')
 
            model_predictions_hex_lines = model_predictions_hex_file.readlines()
  
            model_predictions_hex_one = eval(model_predictions_hex_lines[0])[:-1]
            
            read_PR(test_labels, model_predictions_hex_one, lines_dict)
            read_ROC(test_labels, model_predictions_hex_one, lines_dict)

    print(some_path)
    for val in vals_in_lines:
        if len(lines_dict[val]) == 0:
            continue 
        print(val.replace(" = ", "") + arrayToTable(lines_dict[val], True, True, False))

for some_path in paths:

    lines_dict = dict()
    sd_dict = dict()

    for val in vals_in_lines:
        lines_dict[val] = [] 
        sd_dict[val] = []   

    for seed in seed_list:
        model_predictions_hex = []
        setSeed(seed)

        for number in range(1, NUM_TESTS + 1): 
            model_predictions_hex_file = open(hex_predictions_name(some_path, number, 'weak', 1), 'r')
 
            model_predictions_hex_lines = model_predictions_hex_file.readlines()
  
            model_predictions_hex_one = eval(model_predictions_hex_lines[0])[:-1]

            for x in model_predictions_hex_one:
                model_predictions_hex.append(x)
            
        read_PR(test_labels_long, model_predictions_hex, lines_dict)
        read_ROC(test_labels_long, model_predictions_hex, lines_dict)

    print(some_path)
    for val in vals_in_lines:
        if len(lines_dict[val]) == 0:
            continue 
        print(val.replace(" = ", "") + arrayToTable(lines_dict[val], True, True, False))