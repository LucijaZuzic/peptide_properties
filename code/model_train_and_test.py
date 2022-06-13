#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from after_training import after_training, model_predict
from automate_training import load_data_SA, model_training, average_model
from utils import MODEL_DATA_PATH 
from custom_plots import make_ROC_plots, make_PR_plots, output_metrics, hist_predicted, decorate_stats
from sklearn.model_selection import StratifiedKFold
from automate_training import data_and_labels_from_indices
import numpy as np

# Algorithm settings 
N_FOLDS_FIRST = 5
N_FOLDS_SECOND = 5
EPOCHS = 30

# Load the SA and NSA data
# Choose loading AP, APH, logP or polarity
function = "selu"
data_to_load = "AP"

if len(sys.argv) > 0 and sys.argv[1] in {"AP", "APH", "logP", "polarity_selu", "polarity_relu" }:
    data_to_load = sys.argv[1] 
    
if data_to_load == "polarity_relu":
    function = "relu"
    
SA, NSA = load_data_SA(data_to_load)

# Calculate weight factor for NSA peptides.
# In our data, there are more peptides that do exhibit self assembly property than are those that do not. Therefore,
# during model training, we must adjust weight factors to combat this data imbalance.
factor_NSA = len(SA) / len(NSA)

# Define random seed
seed = 42

# Define N-fold cross validation test harness for splitting the test data from the train and validation data
kfold_first = StratifiedKFold(n_splits=N_FOLDS_FIRST, shuffle=True, random_state=seed)
# Define N-fold cross validation test harness for splitting the validation from the train data
kfold_second = StratifiedKFold(n_splits=N_FOLDS_SECOND, shuffle=True, random_state=seed) 

# Merge SA nad NSA data the train and validation subsets.
all_data = SA + NSA
all_labels = np.ones(len(SA) + len(NSA))
all_labels[len(SA):] *= 0
test_number = 0
 
for train_and_validation_data_indices, test_data_indices in kfold_first.split(all_data, all_labels):
    test_number += 1
    # Write output to file
    sys.stdout = open(MODEL_DATA_PATH[data_to_load]+str(test_number)+"_training_log_"+data_to_load+".txt", "w", encoding="utf-8")
    
    model_type = 0
    # Convert train and validation indices to train and validation data and train and validation labels
    train_and_validation_data, train_and_validation_labels = data_and_labels_from_indices(all_data, all_labels, train_and_validation_data_indices, model_type)
    
    # Convert test indices to test data and test labels
    test_data, test_labels = data_and_labels_from_indices(all_data, all_labels, test_data_indices, model_type)
        
    # Train the aminoacid model
    best_model_index_amino, history_amino = model_training(test_number, train_and_validation_data, train_and_validation_labels, kfold_second, EPOCHS, factor_NSA, model_type=0, data_to_load=data_to_load, function=function)
    best_model_file_amino = MODEL_DATA_PATH[data_to_load]+str(test_number)+'_best_model_amino_'+data_to_load+'_'+str(best_model_index_amino)+'.h5'
    print(best_model_file_amino)
     
    model_type = 1
    # Convert train and validation indices to train and validation data and train and validation labels
    train_and_validation_data, train_and_validation_labels = data_and_labels_from_indices(all_data, all_labels, train_and_validation_data_indices, model_type)
    
    # Convert test indices to test data and test labels
    test_data, test_labels = data_and_labels_from_indices(all_data, all_labels, test_data_indices, model_type)
    
    # Train the dipeptide model
    best_model_index_di, history_di = model_training(test_number, train_and_validation_data, train_and_validation_labels, kfold_second, EPOCHS, factor_NSA, model_type=1, data_to_load=data_to_load, function=function)
    best_model_file_di = MODEL_DATA_PATH[data_to_load]+str(test_number)+'_best_model_di_'+data_to_load+'_'+str(best_model_index_di)+'.h5'
    print(best_model_file_di)
    
    model_type = 2
    # Convert train and validation indices to train and validation data and train and validation labels
    train_and_validation_data, train_and_validation_labels = data_and_labels_from_indices(all_data, all_labels, train_and_validation_data_indices, model_type)
    
    # Convert test indices to test data and test labels
    test_data, test_labels = data_and_labels_from_indices(all_data, all_labels, test_data_indices, model_type)
    
    # Train the tripeptide model
    best_model_index_tri, history_tri = model_training(test_number, train_and_validation_data, train_and_validation_labels, kfold_second, EPOCHS, factor_NSA, model_type=2, data_to_load=data_to_load, function=function)
    best_model_file_tri = MODEL_DATA_PATH[data_to_load]+str(test_number)+'_best_model_tri_'+data_to_load+'_'+str(best_model_index_tri)+'.h5'
    print(best_model_file_tri)
    
    model_type=-1
    # Convert train and validation indices to train and validation data and train and validation labels
    train_and_validation_data, train_and_validation_labels = data_and_labels_from_indices(all_data, all_labels, train_and_validation_data_indices, model_type)
    
    # Convert test indices to test data and test labels
    test_data, test_labels = data_and_labels_from_indices(all_data, all_labels, test_data_indices, model_type)
    
    # Train the ansamble model
    best_model_index_ansamble, history_ansamble = model_training(test_number, train_and_validation_data, train_and_validation_labels, kfold_second, EPOCHS, factor_NSA, data_to_load=data_to_load, function=function)
    best_model_file_ansamble = MODEL_DATA_PATH[data_to_load]+str(test_number)+'_best_model_ansamble_'+data_to_load+'_'+str(best_model_index_ansamble)+'.h5'
    print(best_model_file_ansamble)
    
    # Get predictions from all the models for data that was labeled beforehand
    model_predictions_amino = model_predict(test_data, test_labels, best_model_file_amino, model_type=0, no_model_type = False)
    model_predictions_di = model_predict(test_data, test_labels, best_model_file_di, model_type=1, no_model_type = False)
    model_predictions_tri = model_predict(test_data, test_labels, best_model_file_tri, model_type=2, no_model_type = False)
    model_predictions_ansamble = model_predict(test_data, test_labels, best_model_file_ansamble, no_model_type = False)
    
    # Use the results from the aminoacid, dipeptide and tripeptide models to generate new predictions
    model_predictions_voting = average_model(model_predictions_amino, model_predictions_di, model_predictions_tri, use_binary=True)
    model_predictions_avg = average_model(model_predictions_amino, model_predictions_di, model_predictions_tri)
    
    #Plot ROC curves for all models
    make_ROC_plots(test_number, test_labels, 
                    'ROC_curve', 
                    data_to_load=data_to_load,
                    merge=False,
                    model_predictions_amino=model_predictions_amino, 
                    model_predictions_di=model_predictions_di, 
                    model_predictions_tri=model_predictions_tri, 
                    model_predictions_ansamble=model_predictions_ansamble,
                    model_predictions_voting=model_predictions_voting,
                    model_predictions_avg=model_predictions_avg)
    
    #Plot PR curves for all models
    make_PR_plots(test_number, test_labels, 
                    'PR_curve', 
                    data_to_load=data_to_load,
                    merge=False,
                    model_predictions_amino=model_predictions_amino, 
                    model_predictions_di=model_predictions_di, 
                    model_predictions_tri=model_predictions_tri, 
                    model_predictions_ansamble=model_predictions_ansamble,
                    model_predictions_voting=model_predictions_voting,
                    model_predictions_avg=model_predictions_avg)
    
    # Output accuracy, validation accuracy, loss and validation loss for all models
    decorate_stats("aminokiselina", best_model_index_amino - 1, history_amino, data_to_load)
    decorate_stats("dipeptida", best_model_index_di - 1, history_di, data_to_load)
    decorate_stats("tripeptida", best_model_index_tri - 1, history_tri, data_to_load)
    decorate_stats("ansambla", best_model_index_ansamble - 1, history_ansamble, data_to_load)
    
    # Output adjusted accuracy, F1 score and ROC AUC score for all models
    output_metrics(test_labels, model_predictions_amino, "aminokiselina", data_to_load)
    output_metrics(test_labels, model_predictions_di, "dipeptida", data_to_load)
    output_metrics(test_labels, model_predictions_tri, "tripeptida", data_to_load)
    output_metrics(test_labels, model_predictions_ansamble, "ansambla", data_to_load)
    output_metrics(test_labels, model_predictions_voting, "glasovanja", data_to_load)
    output_metrics(test_labels, model_predictions_avg, "prosjeka", data_to_load)
    
    # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
    hist_predicted(test_number, test_labels, model_predictions_amino, "aminokiselina", data_to_load)
    hist_predicted(test_number, test_labels, model_predictions_di, "dipeptida", data_to_load)
    hist_predicted(test_number, test_labels, model_predictions_tri, "tripeptida", data_to_load)
    hist_predicted(test_number, test_labels, model_predictions_ansamble, "ansambla", data_to_load)
    hist_predicted(test_number, test_labels, model_predictions_voting, "glasovanja", data_to_load)
    hist_predicted(test_number, test_labels, model_predictions_avg, "prosjeka", data_to_load)
    
    # Close output file
    sys.stdout.close()
    
    # Generate predictions on data that has no label beforehand
    after_training(test_number, best_model_file_amino, best_model_file_di, best_model_file_tri, best_model_file_ansamble, data_to_load1=data_to_load)