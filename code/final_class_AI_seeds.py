import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import DATA_PATH
from custom_plots import my_accuracy_calculate, weird_division, convert_to_binary
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

plt.rcParams.update({'font.size': 22})
PRthr = {'../final_all/human_AI_predict.txt': 0.37450, '../final_seq/human_AI_predict.txt': 0.40009, '../final_AP/human_AI_predict.txt': 0.44563,
        "../final_TSNE_seq/human_AI_predict.txt": 0.41442, "../final_TSNE_AP_seq/human_AI_predict.txt": 0.48019}
ROCthr = {'../final_all/human_AI_predict.txt': 0.74304, '../final_seq/human_AI_predict.txt': 0.66199, '../final_AP/human_AI_predict.txt': 0.68748,
        "../final_TSNE_seq/human_AI_predict.txt": 0.65300, "../final_TSNE_AP_seq/human_AI_predict.txt": 0.67038}

def returnGMEAN(actual, pred):
    tn = 0
    tp = 0
    apo = 0
    ane = 0
    for i in range(len(pred)):
        a = actual[i]
        p = pred[i]
        if a == 1:
            apo += 1
        else:
            ane += 1
        if p == a:
            if a == 1:
                tp += 1
            else:
                tn += 1
    
    return np.sqrt(tp / apo * tn / ane)

def read_ROC(test_labels, model_predictions, lines_dict, thrPR, thrROC, name, seed): 
    # Get false positive rate and true positive rate.
    fpr, tpr, thresholds = roc_curve(test_labels, model_predictions) 

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ix = np.argmax(gmeans) 

    # Get recall and precision.
    precision, recall, thresholdsPR = precision_recall_curve(
        test_labels, model_predictions
    ) 

    # Calculate the F1 score for each threshold
    fscore = []
    for i in range(len(precision)):
        fscore.append(
            weird_division(2 * precision[i] * recall[i], precision[i] + recall[i])
        )

    # Locate the index of the largest F1 score
    ixPR = np.argmax(fscore)

    model_predictions_binary_thrPR_old = convert_to_binary(model_predictions, thrPR)
    model_predictions_binary_thrPR_new = convert_to_binary(model_predictions, thresholdsPR[ixPR])
    model_predictions_binary_thrROC_old = convert_to_binary(model_predictions, thrROC)
    model_predictions_binary_thrROC_new = convert_to_binary(model_predictions, thresholds[ix])

    model_predictions_binary = convert_to_binary(model_predictions, 0.5)

    lines_dict['ROC thr old = '].append(thrROC)
    lines_dict['ROC thr new = '].append(thresholds[ix]) 
    lines_dict['ROC AUC = '].append(roc_auc_score(test_labels, model_predictions))
    lines_dict['gmean (0.5) = '].append(returnGMEAN(test_labels, model_predictions_binary))
    lines_dict['gmean (PR thr old) = '].append(returnGMEAN(test_labels, model_predictions_binary_thrPR_old))
    lines_dict['gmean (ROC thr old) = '].append(returnGMEAN(test_labels, model_predictions_binary_thrROC_old))
    lines_dict['gmean (PR thr new) = '].append(returnGMEAN(test_labels, model_predictions_binary_thrPR_new))
    lines_dict['gmean (ROC thr new) = '].append(returnGMEAN(test_labels, model_predictions_binary_thrROC_new))
    #lines_dict['gmean new = '].append(gmeans[ix]) 
    lines_dict['Accuracy (ROC thr old) = '].append(my_accuracy_calculate(test_labels, model_predictions, thrROC))
    lines_dict['Accuracy (ROC thr new) = '].append(my_accuracy_calculate(test_labels, model_predictions, thresholds[ix]))
     
    plt.figure()
    #plt.title(
    #    name + " model"
    #    + "\nReceiver operating characteristic (ROC) curve"
    #)
 
    #plt.axvline(fpr[ix], linestyle = '--', color = 'y')
    #plt.axhline(tpr[ix],  linestyle = '--', color = 'y')

    #radius = np.sqrt(np.power(fpr[ix], 2) + np.power(1 - tpr[ix], 2))
    #circle1 = plt.Circle((0, 1), radius, color = '#2e85ff')
    fig = plt.gcf()
    ax = fig.gca()
    #ax.add_patch(circle1)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    plt.arrow(fpr[ix], tpr[ix], - fpr[ix], 1 - tpr[ix], length_includes_head = True, head_width = 0.02)
    
    # Plot ROC curve.
    plt.plot(fpr, tpr, "r", label="model performance")
    plt.plot(fpr[ix], tpr[ix], "o", markerfacecolor="r", markeredgecolor="k")

    # Plot random guessing ROC curve.
    plt.plot([0, 1], [0, 1], "c", label="random guessing")

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.savefig(
        '../seeds/all_seeds/' + name + '_ROC_only_AI_' + str(seed) + '.png',
        bbox_inches="tight",
    )

    plt.close()

def read_PR(test_labels, model_predictions, lines_dict, thrPR, thrROC, name, seed):  
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

    # Get false positive rate and true positive rate.
    fpr, tpr, thresholdsROC = roc_curve(test_labels, model_predictions) 

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ixROC = np.argmax(gmeans)  

    model_predictions_binary_thrPR_old = convert_to_binary(model_predictions, thrPR)
    model_predictions_binary_thrPR_new = convert_to_binary(model_predictions, thresholds[ix])
    model_predictions_binary_thrROC_old = convert_to_binary(model_predictions, thrROC)
    model_predictions_binary_thrROC_new = convert_to_binary(model_predictions, thresholdsROC[ixROC])
    model_predictions_binary = convert_to_binary(model_predictions, 0.5)
    
    lines_dict['PR thr old = '].append(thrPR)
    lines_dict['PR thr new = '].append(thresholds[ix])
    lines_dict['PR AUC = '].append(auc(recall, precision))  
    lines_dict['F1 (0.5) = '].append(f1_score(test_labels, model_predictions_binary))
    lines_dict['F1 (PR thr old) = '].append(f1_score(test_labels, model_predictions_binary_thrPR_old))
    lines_dict['F1 (ROC thr old) = '].append(f1_score(test_labels, model_predictions_binary_thrROC_old)) 
    lines_dict['F1 (PR thr new) = '].append(f1_score(test_labels, model_predictions_binary_thrPR_new))
    lines_dict['F1 (ROC thr new) = '].append(f1_score(test_labels, model_predictions_binary_thrROC_new))
    lines_dict['Accuracy (PR thr old) = '].append(my_accuracy_calculate(test_labels, model_predictions_binary_thrPR_old, thrPR))
    lines_dict['Accuracy (PR thr new) = '].append(my_accuracy_calculate(test_labels, model_predictions_binary_thrPR_new, thresholds[ix]))
    lines_dict['Accuracy (0.5) = '].append(my_accuracy_calculate(test_labels, model_predictions, 0.5))
    
    plt.figure()
    #plt.title(
    #    name + " model"
    #    + "\nPrecision - Recall (PR) curve"
    #)  

    #plt.axvline(recall[ix], linestyle = '--', color = 'y')
    #plt.axhline(precision[ix],  linestyle = '--', color = 'y')

    #radius = np.sqrt(np.power(1 - recall[ix], 2) + np.power(1 - precision[ix], 2))
    #circle1 = plt.Circle((1, 1), radius, color = '#2e85ff')
    fig = plt.gcf()
    ax = fig.gca()
    #ax.add_patch(circle1)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    plt.arrow(recall[ix], precision[ix], 1 - recall[ix], 1 - precision[ix], length_includes_head = True, head_width = 0.02)
    
    # Plot PR curve.
    plt.plot(recall, precision, "r", label="model performance")
    plt.plot(
        recall[ix], precision[ix], "o", markerfacecolor="r", markeredgecolor="k"
    )
 
    # Calculate the no skill line as the proportion of the positive class
    num_positive = 0
    for value in test_labels:
        if value == 1:
            num_positive += 1
    no_skill = num_positive / len(test_labels)

    # Plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], "c", label="random guessing")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.savefig(
        '../seeds/all_seeds/' + name + '_PR_only_AI_' + str(seed) + '.png',
        bbox_inches="tight",
    )

    plt.close()

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

df = pd.read_csv(DATA_PATH + "human_AI.csv", sep = ";")

dict_human_AI = {}
for i in range(len(df['pep'])):
    if df['expert'][i] == 'Human':
        continue
    dict_human_AI[df['pep'][i]] = str(df['agg'][i])

seq_example = ''
for i in range(24):
    seq_example += 'A'
dict_human_AI[seq_example] = '1' 

best_batch_size = 600
best_model = ''  
NUM_TESTS = 5

offset = 1 
  
properties = np.ones(95)
masking_value = 2

actual_AP_all = []
actual_AP = []
actual_AP_long = []
test_labels = []
test_labels_long = []
thold_AP = []
thold_AP_long = []

for i in range(len(df['AP'])):
    actual_AP_all.append(df['AP'][i]) 
    if df['expert'][i] == 'Human':
        continue
    actual_AP.append(df['AP'][i]) 

threshold = np.mean(actual_AP_all)
 
for i in range(len(df['AP'])):
    if df['expert'][i] == 'Human':
        continue
    test_labels.append(int(df['agg'][i]))  
    if df['AP'][i] < threshold:
        thold_AP.append(0) 
    else:
        thold_AP.append(1) 

print("Mean:", np.mean(actual_AP), "Mod:", np.argmax(np.bincount(actual_AP)), "StD:", np.std(actual_AP), "Var:", np.var(actual_AP))
print("Min:", np.min(actual_AP), "Q1:", np.quantile(actual_AP, .25), "Median:", np.median(actual_AP), "Q2:", np.quantile(actual_AP, .75), "Max:", np.max(actual_AP))

for number in range(1, NUM_TESTS + 1):  
    for i in range(len(df['AP'])):
        if df['expert'][i] == 'Human':
            continue
        actual_AP_long.append(i) 
        test_labels_long.append(int(df['agg'][i])) 
        if df['AP'][i] < threshold:
            thold_AP_long.append(0) 
        else:
            thold_AP_long.append(1)  
            
if not os.path.exists("../seeds/all_seeds/"):
    os.makedirs("../seeds/all_seeds/")
 
paths = ["../final_AP/human_AI_predict.txt", "../final_seq/human_AI_predict.txt", "../final_all/human_AI_predict.txt", 
        "../final_TSNE_seq/human_AI_predict.txt", "../final_TSNE_AP_seq/human_AI_predict.txt"] 
names = ["AP", "SP", "Hybrid AP-SP",  "t-SNE SP", "t-SNE AP-SP"]  

header_line = "Metric"
for i in names:
    header_line += ";" + i

vals_in_lines = [ 
'ROC thr old = ', 'PR thr old = ', 
'ROC AUC = ', 'gmean (ROC thr old) = ', 'F1 (ROC thr old) = ', 'Accuracy (ROC thr old) = ', 
'PR AUC = ', 'gmean (PR thr old) = ', 'F1 (PR thr old) = ', 'Accuracy (PR thr old) = ', 
'gmean (0.5) = ', 'F1 (0.5) = ', 'Accuracy (0.5) = ',  
'ROC thr new = ', 'PR thr new = ', 
'gmean (ROC thr new) = ', 'F1 (ROC thr new) = ', 'Accuracy (ROC thr new) = ', 
'gmean (PR thr new) = ', 'F1 (PR thr new) = ', 'Accuracy (PR thr new) = ', # 'gmean new = '
 ]

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276] 

ress = []
resnew = []

for seed in seed_list:
    print(seed)

    lines_dict = dict()
    sd_dict = dict()
    for val in vals_in_lines:
        lines_dict[val] = [] 
        sd_dict[val] = []  

    ind = -1
    for some_path in paths: 
        ind += 1

        df = pd.read_csv(DATA_PATH + "AI_results_folds_" + str(seed) + ".csv", sep = ";")
        model_predictions_human_one = df[names[ind]]
        
        read_PR(test_labels, model_predictions_human_one, lines_dict, PRthr[some_path], ROCthr[some_path], names[ind], seed)
        read_ROC(test_labels, model_predictions_human_one, lines_dict, PRthr[some_path], ROCthr[some_path], names[ind], seed)
    
    print(header_line)
    ress.append(header_line + "\n")
    for val in vals_in_lines:
        if val.find('new') != -1:
            continue
        if len(lines_dict[val]) == 0:
            continue 
        ress[-1] += val.replace(" = ", "") + arrayToTable(lines_dict[val], True, True, False).replace(" \\\\", "\n").replace(" & ", ";")
        print(val.replace(" = ", "") + arrayToTable(lines_dict[val], True, True, False).replace(" \\\\", "").replace(" & ", ";"))

    save_ress = open(DATA_PATH + "final_class_AI_" + str(seed) + ".csv", "w")
    save_ress.write(ress[-1])
    save_ress.close()
 
    ressss = ress[-1].split("\n")[1:-1]
    row_titles = ress[-1].split("\n")[1:-1]

    for j in range(len(ressss)):
        ressss[j] = ressss[j].split(";")[1:]
        row_titles[j] = row_titles[j].split(";")[0]
        for k in range(len(ressss[j])):
            ressss[j][k] = float(ressss[j][k])
 
    resnew.append(ressss) 

finalres = [[0 for k in range(len(resnew[0][0]))] for j in range(len(resnew[0]))]
for i in range(len(resnew)):
    for j in range(len(resnew[0])):
        for k in range(len(resnew[0][0])):
            finalres[j][k] += resnew[i][j][k]
for j in range(len(resnew[0])):
    for k in range(len(resnew[0][0])):
        finalres[j][k] /= len(resnew)

finalresstring = header_line + "\n"
for j in range(len(resnew[0])):
    finalresstring += row_titles[j]
    for k in range(len(resnew[0][0])):
        finalresstring += ";" + str(finalres[j][k] )
    finalresstring += "\n"
print(finalresstring)

save_ress = open(DATA_PATH + "final_class_AI_avg.csv", "w")
save_ress.write(ress[-1])
save_ress.close()
