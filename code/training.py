import numpy as np
from automate_training import load_data_SA, model_training, merge_data, data_and_labels_from_indices 
from utils import DATA_PATH, MODEL_DATA_PATH
import sys 
from sklearn.model_selection import StratifiedKFold

# Algorithm settings 
N_FOLDS_FIRST = 2
N_FOLDS_SECOND = 2
EPOCHS = 70
names = ['AP']
offset = 1
# Define random seed
seed = 42
SA_data = np.load(DATA_PATH+'data_SA.npy')
  
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

# Define N-fold cross validation test harness for splitting the test data from the train and validation data
kfold_first = StratifiedKFold(n_splits=N_FOLDS_FIRST, shuffle=True, random_state=seed)
# Define N-fold cross validation test harness for splitting the validation from the train data
kfold_second = StratifiedKFold(n_splits=N_FOLDS_SECOND, shuffle=True, random_state=seed) 
 
test_number = 0

for train_and_validation_data_indices, test_data_indices in kfold_first.split(all_data, all_labels):
    test_number += 1
      
    model_type = -1
    # Convert train and validation indices to train and validation data and train and validation labels
    train_and_validation_data, train_and_validation_labels = data_and_labels_from_indices(all_data, all_labels, train_and_validation_data_indices) 
    
    # Convert test indices to test data and test labels
    test_data, test_labels = data_and_labels_from_indices(all_data, all_labels, test_data_indices)

    #train_and_validation_data, test_data, train_and_validation_labels, test_labels = train_test_split(all_data, all_labels, test_size= 1 / N_FOLDS_FIRST, random_state=seed, stratify = all_labels)
 
    # Write output to file
    sys.stdout = open(MODEL_DATA_PATH+str(test_number)+"_training_log_multiple_properties.txt", "w", encoding="utf-8")
    
    # Train the ansamble model
    model_training(num_props, test_number, train_and_validation_data, train_and_validation_labels, kfold_second, EPOCHS, factor_NSA, test_data, test_labels, properties, names, offset, mask_value=masking_value)
         
    # Close output file
    sys.stdout.close()