from utils import predictions_name, DATA_PATH, SEQ_MODEL_DATA_PATH, MODEL_DATA_PATH, MY_MODEL_DATA_PATH, setSeed, PATH_TO_EXTENSION
from custom_plots import merge_type_iteration, results_name
import matplotlib.pyplot as plt
import os
import numpy as np

def read_one_prediction(some_path, test_number, final_model_type, iteration):
    file = open(predictions_name(some_path, test_number, final_model_type, iteration), "r")
    lines = file.readlines()
    predictions = eval(lines[0])
    labels = eval(lines[1])
    file.close()
    return predictions, labels 

def read_all_model_predictions(some_path, min_test_number, max_test_number, final_model_type, iteration):
    all_predictions = []
    all_labels = []
    for test_number in range(min_test_number, max_test_number + 1): 
        predictions, labels = read_one_prediction(some_path, test_number, final_model_type, iteration)
        for prediction in predictions:
            all_predictions.append(prediction)
        for label in labels:
            all_labels.append(label) 
    return all_predictions, all_labels

def hist_predicted_multiple(
    model_type,
    test_number,
    final_model_type,
    iteration,
    test_labels,
    model_predictions,
    seeds,
    save
):
    # Create a histogram of the predicted probabilities only for the peptides that show self-assembly
    plot_label = []
    for seed in seeds:
        plot_label.append("Seed " + str(seed))

    model_predictions_true = []
    for model_num in range(len(model_predictions)):
        model_predictions_true.append([])
        for x in range(len(test_labels[model_num])):
            if test_labels[model_num][x] == 1.0:
                model_predictions_true[-1].append(float(model_predictions[model_num][x]))

    plt.figure()
    plt.title(
        merge_type_iteration(model_type, final_model_type, iteration, test_number).replace("Test 0 Weak 1", "All tests")
        + "\nHistogram of predicted self assembly probability\nfor peptides with self assembly"
    )
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides")
    plt.hist(
        model_predictions_true, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], label=plot_label
    ) 
    plt.legend()
    plt.savefig(save + "_SA.png")
    plt.close()

    # Create a histogram of the predicted probabilities only for the peptides that don't show self-assembly

    model_predictions_false = []
    for model_num in range(len(model_predictions)):
        model_predictions_false.append([])
        for x in range(len(test_labels[model_num])):
            if test_labels[model_num][x] == 0.0:
                model_predictions_false[-1].append(float(model_predictions[model_num][x]))

    plt.figure()
    plt.title(
        merge_type_iteration(model_type, final_model_type, iteration, test_number).replace("Test 0 Weak 1", "All tests")
        + "\nHistogram of predicted self assembly probability\nfor peptides without self assembly"
    )
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides")
    plt.hist(
        model_predictions_false,
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], label=plot_label
    ) 
    plt.legend()
    plt.savefig(save + "_NSA.png")
    plt.close() 

def read_one_result(some_path, test_number):       
    file = open(results_name(some_path, test_number), "r") 
    lines = file.readlines() 
    file.close()
    return lines

def interesting_lines(some_path, test_number, find_str):     
    lines = read_one_result(some_path, test_number) 
    new_lines = []
    for line_num in range(len(lines)):
        if lines[line_num].count(find_str) > 0 and (lines[line_num].count("Weak 1") > 0 or lines[line_num].count("Best params") > 0):   
            new_lines.append(lines[line_num])
    return new_lines

if not os.path.exists("../seeds/all_seeds/"):
    os.makedirs("../seeds/all_seeds/")

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
paths = [SEQ_MODEL_DATA_PATH, MODEL_DATA_PATH, MY_MODEL_DATA_PATH]
NUM_TESTS = 5

for some_path in paths:
    seed_predictions = []
    seed_labels = []
    for seed in seed_list:
        setSeed(seed)
        all_predictions, all_labels = read_all_model_predictions(some_path, 1, 5, "weak", 1)
        seed_predictions.append(all_predictions)
        seed_labels.append(all_labels)
    hist_predicted_multiple(
        some_path,
        0,
        "weak",
        1,
        seed_labels,
        seed_predictions,
        seed_list,
        "../seeds/all_seeds/" + PATH_TO_EXTENSION[some_path] + "_hist_all_seeds"
    )
 
def findValInLine(line, name):
    if line.find(name) == -1:
        return False 
    start = line.find(name) + len(name) 
    isFloat = False 
    end = start
    while (ord(line[end]) >= ord('0') and ord(line[end]) <= ord('9')) or line[end] == '.':
        if line[end] == '.':
            isFloat = True
        end += 1
    if isFloat:
        return float(line[start:end])
    else:
        return int(line[start:end])  

def doubleArrayToTable(array1, array2, addArray, addPercent):
    retstr = "" 
    for i in range(len(array1)): 
        if addArray:
            if not addPercent:
                retstr += " & %.5f (%.5f)" %(array1[i], array2[i])
            else:
                retstr += " & %.5f\\%% (%.5f\\%%)" %(array1[i], array2[i])
    if not addPercent:
        retstr += " & %.5f (%.5f)" %(np.mean(array1), np.mean(array2)) 
    else:
        retstr += " & %.5f\\%% (%.5f\\%%)" %(np.mean(array1), np.mean(array2))  
    return retstr + " \\\\"

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
    most_freq = array[0]
    num_most = freqdict[most_freq]
    for i in freqdict:
        if freqdict[i] > num_most:
            most_freq = i
            num_most = freqdict[i]
    if addAvg:
        retstr += " & %.5f" %(np.mean(array))
    if addMostFrequent:
        retstr += " & %d (%d/%d)" %(most_freq, num_most, len(array)) 
    return retstr + " \\\\"

vals_in_lines = ['num_cells: ', 'kernel_size: ', 'dense: ',
'Maximum accuracy = ', 'Minimal loss = ',
'Accuracy = ', 'Loss = ',
'ROC thr = ','ROC AUC = ', 'gmean = ', 
'PR thr = ', 'PR AUC = ', 'F1 = ', 'F1 (0.5) = ', 'F1 (thr) = ',
'Accuracy (0.5) = ', 'Accuracy (ROC thr) = ', 'Accuracy (PR thr) = ']
 
for some_path in paths:

    lines_dict = dict()
    sd_dict = dict()

    for val in vals_in_lines:
        lines_dict[val] = [] 
        sd_dict[val] = []  

    for seed in [305475974]:
        setSeed(seed)
        interesting_lines_list = []
        for test_number in range(1, NUM_TESTS + 1):
            for line in interesting_lines(some_path, test_number, "Test"):
                interesting_lines_list.append(line)
        for i in range(len(interesting_lines_list)):
            if i == 3 or i == 7:
                continue
            for val in vals_in_lines:
                res = findValInLine(interesting_lines_list[i], val)
                if res != False:
                    lines_dict[val].append(res)
                    if val == 'Loss = ' or val == 'Accuracy = ':
                        index = interesting_lines_list[i].find(val)
                        sd_dict[val].append(findValInLine(interesting_lines_list[i][index:], '('))

    print(some_path)
    for val in vals_in_lines:
        if len(lines_dict[val]) == 0:
            continue
        if val == 'Loss = ':
            print(val.replace(" = ", "") + doubleArrayToTable(lines_dict[val], sd_dict[val], True, False))
            continue
        if val == 'Accuracy = ':
            print(val.replace(" = ", "") + doubleArrayToTable(lines_dict[val], sd_dict[val], True, True))
            continue
        else:
            if val.find(":") == -1:
                print(val.replace(" = ", "") + arrayToTable(lines_dict[val], True, True, False))
            else:
                print(val.replace(": ", "") + arrayToTable(lines_dict[val], True, False, True))

