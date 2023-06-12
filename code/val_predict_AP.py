import numpy as np
from automate_training import load_data_SA_AP, model_predict_AP, data_and_labels_from_indices, reshape_AP, merge_data_AP
from utils import getSeed, DATA_PATH, MY_MODEL_DATA_PATH
import sys 
from sklearn.model_selection import StratifiedKFold
import os 

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
SA, NSA = load_data_SA_AP(SA_data, names, offset, properties, masking_value)

# Calculate weight factor for NSA peptides.
# In our data, there are more peptides that do exhibit self assembly property than are those that do not. Therefore,
# during model training, we must adjust weight factors to combat this data imbalance.
factor_NSA = len(SA) / len(NSA)
 
# Merge SA nad NSA data the train and validation subsets.
all_data, all_labels = merge_data_AP(SA, NSA) 
num_props= len(names) * 3
 
for seed in seed_list:
	
	test_number = 0 

	# Define N-fold cross validation test harness for splitting the test data from the train and validation data
	kfold_first = StratifiedKFold(n_splits=N_FOLDS_FIRST, shuffle=True, random_state=seed)
	# Define N-fold cross validation test harness for splitting the validation from the train data
	kfold_second = StratifiedKFold(n_splits=N_FOLDS_SECOND, shuffle=True, random_state=seed) 
	
	for train_and_validation_data_indices, test_data_indices in kfold_first.split(all_data, all_labels):
		test_number += 1
	       
		# Convert train and validation indices to train and validation data and train and validation labels
		train_and_validation_data, train_and_validation_labels = data_and_labels_from_indices(all_data, all_labels, train_and_validation_data_indices) 
	    
		# Convert test indices to test data and test labels
		test_data, test_labels = data_and_labels_from_indices(all_data, all_labels, test_data_indices)

		#train_and_validation_data, test_data, train_and_validation_labels, test_labels = train_test_split(all_data, all_labels, test_size= 1 / N_FOLDS_FIRST, random_state=seed, stratify = all_labels)
		
		new_path = "../seeds/seed_" + str(seed) + "/only_my_model_data/AP_test_" + str(test_number) + "/"
		  
		#python program to check if a path exists
		#if it doesn’t exist we create one
		if not os.path.exists(new_path):
			os.makedirs(new_path)
			
		indices = []
		for train_data_indices, validation_data_indices in kfold_second.split(train_and_validation_data, train_and_validation_labels): 
			indices.append([train_data_indices, validation_data_indices])
		            
		for best_params_nr in range(1, 4):
			fold_nr = 0 
			for pair in indices:  
			
				fold_nr += 1  
				base = new_path + "AP_test_" + str(test_number) + "_rnn_params_" + str(best_params_nr) + "_fold_" + str(fold_nr)
				best_model_file, best_model_image  = base + ".h5", base + ".png"
				predictions_file_name =  base + "_val_pred.txt"
				best_model = ""
		    		
				train_data_indices = pair[0]
		    
				validation_data_indices = pair[1] 

				# Convert train indices to train data and train labels
				train_data, train_labels = data_and_labels_from_indices(train_and_validation_data, train_and_validation_labels, train_data_indices)
				            
				#train_data, train_labels = reshape_AP(num_props, train_data, train_labels)
				            
				# Convert validation indices to validation data and validation labels
				val_data, val_labels = data_and_labels_from_indices(train_and_validation_data, train_and_validation_labels, [validation_data_indices[-1]])
				            
				#val_data, val_labels = reshape(num_props, val_data, val_labels)
		       
				predictions_file = open(predictions_file_name, "r")
				predictions_file_lines = eval(predictions_file.readlines()[0].replace("\n", ""))
				predictions_file.close()
		      
				model_predictions_peptides = model_predict_AP(num_props, best_batch_size, val_data, val_labels, best_model_file, best_model)
				
				print(model_predictions_peptides) 
				print(predictions_file_lines)
				if len(predictions_file_lines) < len(validation_data_indices):
					predictions_file_lines.append(model_predictions_peptides[0])
				print(predictions_file_lines)
				
				predictions_file = open(predictions_file_name, "w")
				predictions_file.write(str(predictions_file_lines))
				predictions_file.close()
