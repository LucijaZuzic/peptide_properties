import numpy as np
from automate_training import load_data_SA, model_predict, data_and_labels_from_indices, reshape, merge_data
from custom_plots import my_accuracy_calculate, weird_division, convert_to_binary
from utils import getSeed, DATA_PATH, MODEL_DATA_PATH
import sys 
from sklearn.model_selection import StratifiedKFold
import os 
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    auc,
    f1_score,
)

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

def read_ROC(test_labels, model_predictions, lines_dict): 
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
 
    model_predictions_binary_thrPR_new = convert_to_binary(model_predictions, thresholdsPR[ixPR]) 
    model_predictions_binary_thrROC_new = convert_to_binary(model_predictions, thresholds[ix])

    model_predictions_binary = convert_to_binary(model_predictions, 0.5)
 
    lines_dict['ROC thr new = '].append(thresholds[ix]) 
    lines_dict['ROC AUC = '].append(roc_auc_score(test_labels, model_predictions))
    lines_dict['gmean (0.5) = '].append(returnGMEAN(test_labels, model_predictions_binary)) 
    lines_dict['gmean (PR thr new) = '].append(returnGMEAN(test_labels, model_predictions_binary_thrPR_new))
    lines_dict['gmean (ROC thr new) = '].append(returnGMEAN(test_labels, model_predictions_binary_thrROC_new))
    #lines_dict['gmean new = '].append(gmeans[ix])  
    lines_dict['Accuracy (ROC thr new) = '].append(my_accuracy_calculate(test_labels, model_predictions, thresholds[ix])) 

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

    # Get false positive rate and true positive rate.
    fpr, tpr, thresholdsROC = roc_curve(test_labels, model_predictions) 

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ixROC = np.argmax(gmeans)  

    model_predictions_binary_thrPR_new = convert_to_binary(model_predictions, thresholds[ix])
    model_predictions_binary_thrROC_new = convert_to_binary(model_predictions, thresholdsROC[ixROC])
    model_predictions_binary = convert_to_binary(model_predictions, 0.5)
    
    lines_dict['PR thr new = '].append(thresholds[ix])
    lines_dict['PR AUC = '].append(auc(recall, precision))  
    lines_dict['F1 (0.5) = '].append(f1_score(test_labels, model_predictions_binary))
    lines_dict['F1 (PR thr new) = '].append(f1_score(test_labels, model_predictions_binary_thrPR_new))
    lines_dict['F1 (ROC thr new) = '].append(f1_score(test_labels, model_predictions_binary_thrROC_new))
    lines_dict['Accuracy (PR thr new) = '].append(my_accuracy_calculate(test_labels, model_predictions_binary_thrPR_new, thresholds[ix]))
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
'ROC AUC = ', #'gmean (ROC thr old) = ', 'F1 (ROC thr old) = ', 'Accuracy (ROC thr old) = ', 
'PR AUC = ', #'gmean (PR thr old) = ', 'F1 (PR thr old) = ', 'Accuracy (PR thr old) = ', 
'gmean (0.5) = ', 'F1 (0.5) = ', 'Accuracy (0.5) = ',  
'ROC thr new = ', 'PR thr new = ', 
'gmean (ROC thr new) = ', 'F1 (ROC thr new) = ', 'Accuracy (ROC thr new) = ', 
'gmean (PR thr new) = ', 'F1 (PR thr new) = ', 'Accuracy (PR thr new) = ', # 'gmean new = '
 ]
 
only_best_params = True
best_params_ind = [[3, 1, 1, 9, 5], [3, 1, 1, 9, 5], [3, 1, 1, 9, 5], [3, 1, 1, 9, 5], [3, 1, 1, 9, 5]]
 
for typs in range(5):

	lines_dict_typ = dict() 
	for val in vals_in_lines:
		lines_dict_typ[val] = []  
		
	lines_dict_typ_avg = dict() 
	for val in vals_in_lines:
		lines_dict_typ_avg[val] = []  
		
	lines_dict_typ_avg_avg = dict() 
	for val in vals_in_lines:
		lines_dict_typ_avg_avg[val] = []  
		
	lines_dict_typ_avg_avg_avg = dict() 
	for val in vals_in_lines:
		lines_dict_typ_avg_avg_avg[val] = [] 
		 
	lines_dict_typ_avg_avg_avg_avg = dict() 
	for val in vals_in_lines:
		lines_dict_typ_avg_avg_avg_avg[val] = []  
			 
	typlab = []
	typpred = []
	
	seed_nr = 0

	for seed in seed_list:

		# Define N-fold cross validation test harness for splitting the test data from the train and validation data
		kfold_first = StratifiedKFold(n_splits=N_FOLDS_FIRST, shuffle=True, random_state=seed)
		# Define N-fold cross validation test harness for splitting the validation from the train data
		kfold_second = StratifiedKFold(n_splits=N_FOLDS_SECOND, shuffle=True, random_state=seed) 
			
		lines_dict_seed = dict() 
		for val in vals_in_lines:
			lines_dict_seed[val] = []  
			
		lines_dict_seed_avg = dict() 
		for val in vals_in_lines:
			lines_dict_seed_avg[val] = []  
			
		lines_dict_seed_avg_avg = dict() 
		for val in vals_in_lines:
			lines_dict_seed_avg_avg[val] = []  
			
		lines_dict_seed_avg_avg_avg = dict() 
		for val in vals_in_lines:
			lines_dict_seed_avg_avg_avg[val] = []  
				 
		seedlab = []
		seedpred = []
				
		maxpred = 10
		if typs == 0:
			maxpred = 4
		
		for best_params_nr in range(1, maxpred):
			
			if only_best_params and best_params_ind[seed_nr][typs] != best_params_nr:
				continue

			test_number = 0 
			
			lines_dict_param = dict() 
			for val in vals_in_lines:
				lines_dict_param[val] = [] 
				
			lines_dict_param_avg = dict() 
			for val in vals_in_lines:
				lines_dict_param_avg[val] = [] 
				
			lines_dict_param_avg_avg = dict() 
			for val in vals_in_lines:
				lines_dict_param_avg_avg[val] = []   
				 
			paramlab = []
			parampred = []

			for train_and_validation_data_indices, test_data_indices in kfold_first.split(all_data, all_labels):
				test_number += 1
			       
				# Convert train and validation indices to train and validation data and train and validation labels
				train_and_validation_data, train_and_validation_labels = data_and_labels_from_indices(all_data, all_labels, train_and_validation_data_indices) 
			    
				# Convert test indices to test data and test labels
				test_data, test_labels = data_and_labels_from_indices(all_data, all_labels, test_data_indices)

				#train_and_validation_data, test_data, train_and_validation_labels, test_labels = train_test_split(all_data, all_labels, test_size= 1 / N_FOLDS_FIRST, random_state=seed, stratify = all_labels)
				
				indices = []
				 
				lines_dict_test= dict() 
				for val in vals_in_lines:
					lines_dict_test[val] = []  
				
				lines_dict_test_avg = dict() 
				for val in vals_in_lines:
					lines_dict_test_avg[val] = []  
						
				testlab = []
				testpred = []
				
				for train_data_indices, validation_data_indices in kfold_second.split(train_and_validation_data, train_and_validation_labels): 
					indices.append([train_data_indices, validation_data_indices])
					    
				fold_nr = 0 
					
				for pair in indices:    
					 
					lines_dict_fold = dict() 
					for val in vals_in_lines:
						lines_dict_fold[val] = []  
				
					fold_nr += 1  
					
					predictions_file_name_AP = "../seeds/seed_" + str(seed) + "/only_my_model_data/AP_test_" + str(test_number) + "/"   + "AP_test_" + str(test_number) + "_rnn_params_" + str(best_params_nr) + "_fold_" + str(fold_nr) + "_val_pred.txt"
					 
					predictions_file_name_seq = "../seeds/seed_" + str(seed) + "/seq_model_data/seq_test_" + str(test_number) + "/"   + "seq_test_" + str(test_number) + "_rnn_params_" + str(best_params_nr) + "_fold_" + str(fold_nr) + "_val_pred.txt"
					
					predictions_file_name_all = "../seeds/seed_" + str(seed) + "/model_data/all_test_" + str(test_number) + "/"   + "all_test_" + str(test_number) + "_rnn_params_" + str(best_params_nr) + "_fold_" + str(fold_nr) + "_val_pred.txt" 
					
					predictions_file_name_TSNE_seq = "../seeds/seed_" + str(seed) + "/TSNE_seq_model_data/TSNE_seq_test_" + str(test_number) + "/"   + "TSNE_seq_test_" + str(test_number) + "_rnn_params_" + str(best_params_nr) + "_fold_" + str(fold_nr) + "_val_pred.txt"
					
					predictions_file_name_TSNE_ap_seq = "../seeds/seed_" + str(seed) + "/TSNE_ap_seq_model_data/TSNE_ap_seq_test_" + str(test_number) + "/"   + "TSNE_ap_seq_test_" + str(test_number) + "_rnn_params_" + str(best_params_nr) + "_fold_" + str(fold_nr) + "_val_pred.txt"
					
					if typs == 0:
						predictions_file_name_x = predictions_file_name_AP
					if typs == 1:
						predictions_file_name_x = predictions_file_name_seq
					if typs == 2:
						predictions_file_name_x = predictions_file_name_all
					if typs == 3:
						predictions_file_name_x = predictions_file_name_TSNE_seq
					if typs == 4:
						predictions_file_name_x = predictions_file_name_TSNE_ap_seq
			    		
					train_data_indices = pair[0]
			    
					validation_data_indices = pair[1] 

					# Convert train indices to train data and train labels
					train_data, train_labels = data_and_labels_from_indices(train_and_validation_data, train_and_validation_labels, train_data_indices)
						     
					# Convert validation indices to validation data and validation labels
					val_data, val_labels = data_and_labels_from_indices(train_and_validation_data, train_and_validation_labels, validation_data_indices)
						       
					predictions_file_all = open(predictions_file_name_x, "r")
					predictions_file_lines_all = eval(predictions_file_all.readlines()[0].replace("\n", ""))
					predictions_file_all.close()
					 
					read_ROC(val_labels, predictions_file_lines_all, lines_dict_fold)
					read_PR(val_labels, predictions_file_lines_all, lines_dict_fold)
					  
					for x in lines_dict_fold: 
						lines_dict_test_avg[x].append(lines_dict_fold[x][0])
					
					for i in range(len(val_labels)):
						testlab.append(val_labels[i])
						testpred.append(predictions_file_lines_all[i])
						paramlab.append(val_labels[i])
						parampred.append(predictions_file_lines_all[i])
						seedlab.append(val_labels[i])
						seedpred.append(predictions_file_lines_all[i]) 
						typlab.append(val_labels[i])
						typpred.append(predictions_file_lines_all[i]) 
						
				read_ROC(testlab, testpred, lines_dict_test)
				read_PR(testlab, testpred, lines_dict_test)
			 
				for x in lines_dict_test_avg:
					avgval = 0
					for y in lines_dict_test_avg[x]:
						avgval += y
					avgval /= len(lines_dict_test_avg[x])
					lines_dict_test_avg[x] = [avgval] 
				
				for x in lines_dict_test:
					lines_dict_param_avg[x].append(lines_dict_test[x][0])
					
				for x in lines_dict_test_avg:
					lines_dict_param_avg_avg[x].append(lines_dict_test_avg[x][0])
				
			read_ROC(paramlab, parampred, lines_dict_param)
			read_PR(paramlab, parampred, lines_dict_param)   
			
			for x in lines_dict_param_avg:
				avgval = 0
				for y in lines_dict_param_avg[x]:
					avgval += y
				avgval /= len(lines_dict_param_avg[x])
				lines_dict_param_avg[x] = [avgval] 
				
			for x in lines_dict_param_avg_avg:
				avgval = 0
				for y in lines_dict_param_avg_avg[x]:
					avgval += y
				avgval /= len(lines_dict_param_avg_avg[x])
				lines_dict_param_avg_avg[x] = [avgval] 
			
			for x in lines_dict_param:
				lines_dict_seed_avg[x].append(lines_dict_param[x][0])
				
			for x in lines_dict_param_avg:
				lines_dict_seed_avg_avg[x].append(lines_dict_param_avg[x][0])
				
			for x in lines_dict_param_avg_avg:
				lines_dict_seed_avg_avg_avg[x].append(lines_dict_param_avg_avg[x][0])
			
		read_ROC(seedlab, seedpred, lines_dict_seed)
		read_PR(seedlab, seedpred, lines_dict_seed) 
		 
		for x in lines_dict_seed_avg:
			avgval = 0
			for y in lines_dict_seed_avg[x]:
				avgval += y
			avgval /= len(lines_dict_seed_avg[x])
			lines_dict_seed_avg[x] = [avgval] 
			
		for x in lines_dict_seed_avg_avg:
			avgval = 0
			for y in lines_dict_seed_avg_avg[x]:
				avgval += y
			avgval /= len(lines_dict_seed_avg_avg[x])
			lines_dict_seed_avg_avg[x] = [avgval] 
			
		for x in lines_dict_seed_avg_avg_avg:
			avgval = 0
			for y in lines_dict_seed_avg_avg_avg[x]:
				avgval += y
			avgval /= len(lines_dict_seed_avg_avg_avg[x])
			lines_dict_seed_avg_avg_avg[x] = [avgval]
			
		for x in lines_dict_seed:
			lines_dict_typ_avg[x].append(lines_dict_seed[x][0])
			
		for x in lines_dict_seed_avg:
			lines_dict_typ_avg_avg[x].append(lines_dict_seed_avg[x][0])
			
		for x in lines_dict_seed_avg_avg:
			lines_dict_typ_avg_avg_avg[x].append(lines_dict_seed_avg_avg[x][0])
			
		for x in lines_dict_seed_avg_avg_avg:
			lines_dict_typ_avg_avg_avg_avg[x].append(lines_dict_seed_avg_avg_avg[x][0]) 
	
		seed_nr += 1
	
	read_ROC(typlab, typpred, lines_dict_typ)
	read_PR(typlab, typpred, lines_dict_typ) 
	 
	for x in lines_dict_typ_avg:
		avgval = 0
		for y in lines_dict_typ_avg[x]:
			avgval += y
		avgval /= len(lines_dict_typ_avg[x])
		lines_dict_typ_avg[x] = [avgval] 
		
	for x in lines_dict_typ_avg_avg:
		avgval = 0
		for y in lines_dict_typ_avg_avg[x]:
			avgval += y
		avgval /= len(lines_dict_typ_avg_avg[x])
		lines_dict_typ_avg_avg[x] = [avgval] 
		
	for x in lines_dict_typ_avg_avg_avg:
		avgval = 0
		for y in lines_dict_typ_avg_avg_avg[x]:
			avgval += y
		avgval /= len(lines_dict_typ_avg_avg_avg[x])
		lines_dict_typ_avg_avg_avg[x] = [avgval]
		
	for x in lines_dict_typ_avg_avg_avg_avg:
		avgval = 0
		for y in lines_dict_typ_avg_avg_avg_avg[x]:
			avgval += y
		avgval /= len(lines_dict_typ_avg_avg_avg_avg[x])
		lines_dict_typ_avg_avg_avg_avg[x] = [avgval]
	
	print(predictions_file_name_x)
	str1 = ""
	str2 = ""
	for x in lines_dict_typ_avg_avg_avg:
		str1 += x + " "
		if x != "PR thr new = " and x != "ROC thr new = ":
			str2 += str(np.round(lines_dict_typ_avg_avg_avg[x][0], 3)) + " "
		else:
			str2 += str(lines_dict_typ_avg_avg[x][0]) + " "
	print(str1.replace(" thr new)", ")").replace(" = ", ""))
	print(str2)
