import numpy as np
from automate_training import load_data_SA, merge_data
from custom_plots import my_accuracy_calculate, weird_division, convert_to_binary
from utils import getSeed, DATA_PATH, MODEL_DATA_PATH
import pandas as pd
import sys  
import os 
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    auc,
    f1_score,
)


PRthr_old = {'../final_all/human_AI_predict.txt': 0.37450, '../final_seq/human_AI_predict.txt': 0.40009, '../final_AP/human_AI_predict.txt': 0.44563,
        "../final_TSNE_seq/human_AI_predict.txt": 0.41442, "../final_TSNE_AP_seq/human_AI_predict.txt": 0.48019}
ROCthr_old = {'../final_all/human_AI_predict.txt': 0.74304, '../final_seq/human_AI_predict.txt': 0.66199, '../final_AP/human_AI_predict.txt': 0.68748,
        "../final_TSNE_seq/human_AI_predict.txt": 0.65300, "../final_TSNE_AP_seq/human_AI_predict.txt": 0.67038}

PRthr_new = {'../final_all/human_AI_predict.txt': 0.320440828, '../final_seq/human_AI_predict.txt': 0.32140276000000007, '../final_AP/human_AI_predict.txt': 0.22412928599999998,
        "../final_TSNE_seq/human_AI_predict.txt": 0.277013928, "../final_TSNE_AP_seq/human_AI_predict.txt": 0.30854635999999996}
ROCthr_new = {'../final_all/human_AI_predict.txt': 0.6066449939999999, '../final_seq/human_AI_predict.txt': 0.6172134180000001, '../final_AP/human_AI_predict.txt': 0.526199318,
        "../final_TSNE_seq/human_AI_predict.txt": 0.556881612, "../final_TSNE_AP_seq/human_AI_predict.txt": 0.549448272}
 
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

def read_ROC(test_labels, model_predictions, lines_dict, old_PR_thr, new_PR_thr, old_ROC_thr, new_ROC_thr): 
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
    #ixPR = np.argmax(fscore)
 
    model_predictions_binary_thrPR_new = convert_to_binary(model_predictions, new_PR_thr) 
    model_predictions_binary_thrROC_new = convert_to_binary(model_predictions, new_ROC_thr)
    
    model_predictions_binary_thrPR_old = convert_to_binary(model_predictions, old_PR_thr) 
    model_predictions_binary_thrROC_old = convert_to_binary(model_predictions, old_ROC_thr)

    model_predictions_binary = convert_to_binary(model_predictions, 0.5)
 
    #lines_dict['ROC thr new = '].append(thresholds[ix]) 
    lines_dict['ROC AUC = '].append(roc_auc_score(test_labels, model_predictions))
    lines_dict['gmean (0.5) = '].append(returnGMEAN(test_labels, model_predictions_binary)) 
    lines_dict['gmean (PR thr new) = '].append(returnGMEAN(test_labels, model_predictions_binary_thrPR_new))
    lines_dict['gmean (ROC thr new) = '].append(returnGMEAN(test_labels, model_predictions_binary_thrROC_new))
    lines_dict['gmean (PR thr old) = '].append(returnGMEAN(test_labels, model_predictions_binary_thrPR_old))
    lines_dict['gmean (ROC thr old) = '].append(returnGMEAN(test_labels, model_predictions_binary_thrROC_old))
    #lines_dict['gmean new = '].append(gmeans[ix])  
    lines_dict['Accuracy (ROC thr new) = '].append(my_accuracy_calculate(test_labels, model_predictions, new_ROC_thr)) 
    lines_dict['Accuracy (ROC thr old) = '].append(my_accuracy_calculate(test_labels, model_predictions, old_ROC_thr)) 

def read_PR(test_labels, model_predictions, lines_dict, old_PR_thr, new_PR_thr, old_ROC_thr, new_ROC_thr): 
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
    #ix = np.argmax(fscore)

    # Get false positive rate and true positive rate.
    fpr, tpr, thresholdsROC = roc_curve(test_labels, model_predictions) 

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ixROC = np.argmax(gmeans)  

    model_predictions_binary_thrPR_new = convert_to_binary(model_predictions, new_PR_thr) 
    model_predictions_binary_thrROC_new = convert_to_binary(model_predictions, new_ROC_thr)
    
    model_predictions_binary_thrPR_old = convert_to_binary(model_predictions, old_PR_thr) 
    model_predictions_binary_thrROC_old = convert_to_binary(model_predictions, old_ROC_thr)
    
    model_predictions_binary = convert_to_binary(model_predictions, 0.5)
    
    #lines_dict['PR thr new = '].append(thresholds[ix])
    lines_dict['PR AUC = '].append(auc(recall, precision))  
    lines_dict['F1 (0.5) = '].append(f1_score(test_labels, model_predictions_binary))
    lines_dict['F1 (PR thr new) = '].append(f1_score(test_labels, model_predictions_binary_thrPR_new))
    lines_dict['F1 (ROC thr new) = '].append(f1_score(test_labels, model_predictions_binary_thrROC_new))
    lines_dict['F1 (PR thr old) = '].append(f1_score(test_labels, model_predictions_binary_thrPR_old))
    lines_dict['F1 (ROC thr old) = '].append(f1_score(test_labels, model_predictions_binary_thrROC_old))
    lines_dict['Accuracy (PR thr new) = '].append(my_accuracy_calculate(test_labels, model_predictions, new_PR_thr))
    lines_dict['Accuracy (PR thr old) = '].append(my_accuracy_calculate(test_labels, model_predictions, old_PR_thr))
    lines_dict['Accuracy (0.5) = '].append(my_accuracy_calculate(test_labels, model_predictions, 0.5))
    
# Algorithm settings 
N_FOLDS_FIRST = 5
N_FOLDS_SECOND = 5 
names = ['AP']
offset = 1
best_batch_size = 600
# Define random seed
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]  
SA_data = np.load(DATA_PATH+'data_SA_updated.npy', allow_pickle=True).item()
  
properties = np.ones(95)
masking_value = 2
SA, NSA = load_data_SA(SA_data, names, offset, properties, masking_value)

# Calculate weight factor for NSA peptides.
# In our data, there are more peptides that do exhibit self assembly property than are those that do not. Therefore,
# during model training, we must adjust weight factors to combat this data imbalance.
factor_NSA = len(SA) / len(NSA)
 
# Merge SA nad NSA data the train and validation subsets.
all_data, all_labels = merge_data(SA, NSA) 
num_props= len(names) * 3
 
vals_in_lines = [ 
#'ROC thr old = ', 'PR thr old = ', 
'ROC AUC = ', 'gmean (ROC thr old) = ', 'F1 (ROC thr old) = ', 'Accuracy (ROC thr old) = ', 
'PR AUC = ', 'gmean (PR thr old) = ', 'F1 (PR thr old) = ', 'Accuracy (PR thr old) = ', 
'gmean (0.5) = ', 'F1 (0.5) = ', 'Accuracy (0.5) = ',  
#'ROC thr new = ', 'PR thr new = ', 
'gmean (ROC thr new) = ', 'F1 (ROC thr new) = ', 'Accuracy (ROC thr new) = ', 
'gmean (PR thr new) = ', 'F1 (PR thr new) = ', 'Accuracy (PR thr new) = ', # 'gmean new = '
 ]

lines_dict_hex = dict()
for val in vals_in_lines:
	lines_dict_hex[val] = []
		
lines_dict_human_AI = dict()
for val in vals_in_lines:
	lines_dict_human_AI[val] = []
	
lines_dict_human = dict()
for val in vals_in_lines:
	lines_dict_human[val] = []
	
lines_dict_AI = dict()
for val in vals_in_lines:
	lines_dict_AI[val] = []
 	
df = pd.read_csv(DATA_PATH + "41557_2022_1055_MOESM3_ESM_Figure3a_5mer_score_shortMD.csv")
 
actual_AP = [] 

for i in df['AP']:
    actual_AP.append(i) 

threshold = np.mean(actual_AP)

hex_labels = []
for i in df['AP']:
    if i < threshold:
        hex_labels.append(0) 
    else:
        hex_labels.append(1) 
        
df = pd.read_csv(DATA_PATH + "human_AI.csv", sep = ";")
  
human_AI_labels = []
human_labels = []
AI_labels = []

human_index = []
AI_index = []

for i in range(len(df['pep'])): 
	human_AI_labels.append(int(df['agg'][i]))
	if df['expert'][i] == 'Human': 
		human_index.append(i)
		human_labels.append(int(df['agg'][i]))
	else: 
		AI_index.append(i)
		AI_labels.append(int(df['agg'][i]))
    	
paths = ["../final_AP/human_AI_predict.txt", "../final_seq/human_AI_predict.txt", "../final_all/human_AI_predict.txt", 
        "../final_TSNE_seq/human_AI_predict.txt", "../final_TSNE_AP_seq/human_AI_predict.txt"] 
     
for path in paths:
	ind = list(PRthr_old.keys()).index(path)      

	filename = list(PRthr_old.keys())[ind]
	filenamehex = filename.replace("human_AI", "hex")

	filehex = open(filenamehex, "r")
	hexpreds = eval(filehex.readlines()[0])
	filehex.close()
 	
	filehuman = open(filename.replace("final", "final_no_human"), "r")
	humanpreds = eval(filehuman.readlines()[0])
	filehuman.close()
	
	fileAI = open(filename.replace("final", "final_no_AI"), "r")
	AIpreds = eval(fileAI.readlines()[0])
	fileAI.close()
 		 
	humanAIpreds = []
	human_pos = 0
	AI_pos = 0
	for i in range(len(human_AI_labels)):
		if i in human_index:
			humanAIpreds.append(humanpreds[human_pos]) 
			human_pos += 1
		else:
			humanAIpreds.append(AIpreds[AI_pos]) 
			AI_pos += 1
 	
	print(human_labels, humanpreds)
	print(AI_labels, AIpreds)
	print(human_AI_labels, humanAIpreds)
	read_ROC(human_AI_labels, humanAIpreds, lines_dict_human_AI, list(PRthr_old.values())[ind], list(PRthr_new.values())[ind], list(ROCthr_old.values())[ind], list(ROCthr_new.values())[ind])
	read_PR(human_AI_labels, humanAIpreds, lines_dict_human_AI, list(PRthr_old.values())[ind], list(PRthr_new.values())[ind], list(ROCthr_old.values())[ind], list(ROCthr_new.values())[ind])
	
	read_ROC(human_labels, humanpreds, lines_dict_human, list(PRthr_old.values())[ind], list(PRthr_new.values())[ind], list(ROCthr_old.values())[ind], list(ROCthr_new.values())[ind])
	read_PR(human_labels, humanpreds, lines_dict_human, list(PRthr_old.values())[ind], list(PRthr_new.values())[ind], list(ROCthr_old.values())[ind], list(ROCthr_new.values())[ind])
	
	read_ROC(AI_labels, AIpreds, lines_dict_AI, list(PRthr_old.values())[ind], list(PRthr_new.values())[ind], list(ROCthr_old.values())[ind], list(ROCthr_new.values())[ind])
	read_PR(AI_labels, AIpreds, lines_dict_AI, list(PRthr_old.values())[ind], list(PRthr_new.values())[ind], list(ROCthr_old.values())[ind], list(ROCthr_new.values())[ind])
	
	read_ROC(hex_labels, hexpreds, lines_dict_hex, list(PRthr_old.values())[ind], list(PRthr_new.values())[ind], list(ROCthr_old.values())[ind], list(ROCthr_new.values())[ind])
	read_PR(hex_labels, hexpreds, lines_dict_hex, list(PRthr_old.values())[ind], list(PRthr_new.values())[ind], list(ROCthr_old.values())[ind], list(ROCthr_new.values())[ind])
	
header = "val"
for path in paths:
	ind = list(PRthr_old.keys()).index(path)     
	header += " " + list(PRthr_old.keys())[ind].split("/")[1]
 
print("Human") 
print(header) 
for val in lines_dict_human:
	print(val, lines_dict_human[val])
	
print("AI") 
print(header) 
for val in lines_dict_AI:
	print(val, lines_dict_AI[val])
	
print("Human AI") 
print(header) 
for val in lines_dict_human_AI:
	print(val, lines_dict_human_AI[val])
