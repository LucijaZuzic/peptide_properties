# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 05:58:17 2022

@author: Lucija
"""

import numpy as np 
from automate_training import load_data_SA
from utils import DATA_PATH 
import sys 
from after_training import model_predict, after_training_no_model_type
from automate_training import model_training   
from utils import MODEL_DATA_PATH 
from custom_plots import make_ROC_plots, make_PR_plots, output_metrics, hist_predicted, decorate_stats
from sklearn.model_selection import StratifiedKFold   
from automate_training import data_and_labels_from_indices 

# Algorithm settings 
N_FOLDS_FIRST = 5
N_FOLDS_SECOND = 5
EPOCHS = 30

# Load the SA and NSA data
# Choose loading AP, APH, logP or polarity
function = "selu"
data_to_load = "AP_ALL_PROPERTIES"
     
# Calculate weight factor for NSA peptides.
# In our data, there are more peptides that do exhibit self assembly property than are those that do not. Therefore,
# during model training, we must adjust weight factors to combat this data imbalance.

# Define random seed
seed = 42

# Define N-fold cross validation test harness for splitting the test data from the train and validation data
kfold_first = StratifiedKFold(n_splits=N_FOLDS_FIRST, shuffle=True, random_state=seed)
# Define N-fold cross validation test harness for splitting the validation from the train data
kfold_second = StratifiedKFold(n_splits=N_FOLDS_SECOND, shuffle=True, random_state=seed) 
 
SA_data = np.load(DATA_PATH+'data_SA.npy')

test_number = 0
 
SA, NSA = load_data_SA("AP_ALL_PROPERTIES")

# Calculate weight factor for NSA peptides.
# In our data, there are more peptides that do exhibit self assembly property than are those that do not. Therefore,
# during model training, we must adjust weight factors to combat this data imbalance.
factor_NSA = len(SA) / len(NSA)

# Define random seed
seed = 42
 
# Merge SA nad NSA data the train and validation subsets.
all_data = SA + NSA
all_labels = np.ones(len(SA) + len(NSA))
all_labels[len(SA):] *= 0
test_number = 0

for train_and_validation_data_indices, test_data_indices in kfold_first.split(all_data, all_labels):
    test_number += 1
      
    model_type = -1
    # Convert train and validation indices to train and validation data and train and validation labels
    train_and_validation_data, train_and_validation_labels = data_and_labels_from_indices(all_data, all_labels, train_and_validation_data_indices, data_index = -1) 
    
     # Convert test indices to test data and test labels
    test_data, test_labels = data_and_labels_from_indices(all_data, all_labels, test_data_indices, data_index = -1)
     
    # Write output to file
    sys.stdout = open(MODEL_DATA_PATH[data_to_load]+str(test_number)+"_training_log_"+data_to_load+".txt", "w", encoding="utf-8")
    
    # Train the ansamble model
    best_model_index_ansamble, history_ansamble = model_training(test_number, train_and_validation_data, train_and_validation_labels, kfold_second, EPOCHS, factor_NSA, model_type=-1, no_model_type=True, data_to_load = data_to_load,  merge = False, function = function)
    best_model_file_ansamble = MODEL_DATA_PATH[data_to_load]+str(test_number)+'_best_model_ansamble_'+data_to_load+'_'+str(best_model_index_ansamble)+'.h5'
    print(best_model_file_ansamble)
    
    # Get predictions from all the models for data that was labeled beforehand
    model_predictions_ansamble = model_predict(test_data, test_labels, best_model_file_ansamble, model_type=-1, no_model_type = True) 
     
    #Plot ROC curves for all models
    make_ROC_plots(test_number, test_labels, 
                    'ROC_curve', 
                    data_to_load=data_to_load,
                    merge=False,
                    model_predictions_ansamble=model_predictions_ansamble)
    
    #Plot PR curves for all models
    make_PR_plots(test_number, test_labels, 
                    'PR_curve', 
                    data_to_load=data_to_load,
                    merge=False,
                    model_predictions_ansamble=model_predictions_ansamble)
    
    # Output accuracy, validation accuracy, loss and validation loss for all models
    decorate_stats("svih svojstava ansambl", best_model_index_ansamble - 1, history_ansamble, data_to_load) 
    
    # Output adjusted accuracy, F1 score and ROC AUC score for all models
    output_metrics(test_labels, model_predictions_ansamble, "svih svojstava ansambl", data_to_load) 
    
    # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
    hist_predicted(test_number, test_labels, model_predictions_ansamble, "svih svojstava ansambl", data_to_load) 
    
    # Close output file
    sys.stdout.close()
    
    # Generate predictions on data that has no label beforehand
    after_training_no_model_type(test_number, best_model_file_ansamble, data_to_load=data_to_load, model_type=-1, no_model_type = True)