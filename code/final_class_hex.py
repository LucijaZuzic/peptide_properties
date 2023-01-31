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

PRthr = {'../final_AP/hex_predict.txt': 0.37450, '../final_seq/hex_predict.txt': 0.40009, '../final_all/hex_predict.txt': 0.44563}
ROCthr = {'../final_AP/hex_predict.txt': 0.74304, '../final_seq/hex_predict.txt': 0.66199, '../final_all/hex_predict.txt': 0.68748}

def read_ROC(test_labels, model_predictions, lines_dict, thr): 
    # Get false positive rate and true positive rate.
    fpr, tpr, thresholds = roc_curve(test_labels, model_predictions) 

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ix = np.argmax(gmeans) 

    lines_dict['ROC thr old = '].append(thr)
    lines_dict['ROC thr new = '].append(thresholds[ix])
    lines_dict['gmean = '].append(gmeans[ix])
    lines_dict['ROC AUC = '].append(roc_auc_score(test_labels, model_predictions))
    lines_dict['Accuracy (ROC thr old) = '].append(my_accuracy_calculate(test_labels, model_predictions, thr))
    lines_dict['Accuracy (ROC thr new) = '].append(my_accuracy_calculate(test_labels, model_predictions, thresholds[ix]))
   
def read_PR(test_labels, model_predictions, lines_dict, thr):  
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
    model_predictions_binary_thr_old = convert_to_binary(model_predictions, thr)
    model_predictions_binary_thr_new = convert_to_binary(model_predictions, thresholds[ix])
    model_predictions_binary = convert_to_binary(model_predictions, 0.5)
    
    lines_dict['PR thr old = '].append(thr)
    lines_dict['PR thr new = '].append(thresholds[ix])
    lines_dict['PR AUC = '].append(auc(recall, precision)) 
    lines_dict['F1 (0.5) = '].append(f1_score(test_labels, model_predictions_binary))
    lines_dict['F1 (thr old) = '].append(f1_score(test_labels, model_predictions_binary_thr_old))
    lines_dict['F1 (thr new) = '].append(f1_score(test_labels, model_predictions_binary_thr_new))
    lines_dict['Accuracy (PR thr old) = '].append(my_accuracy_calculate(test_labels, model_predictions_binary_thr_old, thr))
    lines_dict['Accuracy (PR thr new) = '].append(my_accuracy_calculate(test_labels, model_predictions_binary_thr_new, thresholds[ix]))
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
    #if addAvg:
        #retstr += " & %.5f" %(np.mean(array))
    if addMostFrequent: 
        most_freq_str = str(sorted(most_freq)[0])
        for i in sorted(most_freq[1:]):
            most_freq_str += ", " + str(i)
        retstr += " & %s (%d/%d)" %(most_freq_str, num_most, len(array)) 
    return retstr + " \\\\" 

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
NUM_TESTS = 5

offset = 1 
  
properties = np.ones(95)
masking_value = 2

actual_AP = []
actual_AP_long = []
test_labels = []
test_labels_long = []

for i in df['AP']:
    actual_AP.append(i) 

threshold = np.mean(actual_AP)
 
for i in df['AP']:
    if i < threshold:
        test_labels.append(0) 
    else:
        test_labels.append(1) 

print("Mean:", np.mean(actual_AP), "Mod:", np.argmax(np.bincount(actual_AP)), "StD:", np.std(actual_AP), "Var:", np.var(actual_AP))
print("Min:", np.min(actual_AP), "Q1:", np.quantile(actual_AP, .25), "Median:", np.median(actual_AP), "Q2:", np.quantile(actual_AP, .75), "Max:", np.max(actual_AP))

for number in range(1, NUM_TESTS + 1): 
    for i in df['AP']:
        actual_AP_long.append(i) 
        if i < threshold:
            test_labels_long.append(0) 
        else:
            test_labels_long.append(1) 
 
if not os.path.exists("../seeds/all_seeds/"):
    os.makedirs("../seeds/all_seeds/")
 
paths = ["../final_seq/hex_predict.txt", "../final_AP/hex_predict.txt", "../final_all/hex_predict.txt",] 

vals_in_lines = [ 
'ROC thr new = ','ROC AUC = ', 'gmean = ', 
'ROC thr old = ',
'PR thr old = ', 'PR thr new = ', 'PR AUC = ', 'F1 (0.5) = ', 'F1 (thr old) = ','F1 (thr new) = ',
'Accuracy (0.5) = ', 'Accuracy (ROC thr new) = ', 'Accuracy (PR thr new) = ', 'Accuracy (ROC thr old) = ', 'Accuracy (PR thr old) = '] 
  
lines_dict = dict()
sd_dict = dict()
for val in vals_in_lines:
    lines_dict[val] = [] 
    sd_dict[val] = []  
for some_path in paths: 
    model_predictions_hex_file = open(some_path, 'r')

    model_predictions_hex_lines = model_predictions_hex_file.readlines()

    model_predictions_hex_one = eval(model_predictions_hex_lines[0])
    
    read_PR(test_labels, model_predictions_hex_one, lines_dict, PRthr[some_path])
    read_ROC(test_labels, model_predictions_hex_one, lines_dict, ROCthr[some_path])

    print(some_path)
for val in vals_in_lines:
    if len(lines_dict[val]) == 0:
        continue 
    print(val.replace(" = ", "") + arrayToTable(lines_dict[val], True, True, False)) 