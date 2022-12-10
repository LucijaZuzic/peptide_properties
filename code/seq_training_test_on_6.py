import numpy as np
from automate_training import load_data_SA_seq, model_training_seq, merge_data_seq, data_and_labels_from_indices, extract_len_from_data_and_labels
from utils import DATA_PATH, SEQ_MODEL_DATA_PATH
import sys 
from sklearn.model_selection import StratifiedKFold

# Algorithm settings 
N_FOLDS_SECOND = 5
EPOCHS = 70
names = ['AP', 'logP', 'APH', 'polarity_selu']
offset = 1
# Define random seed
seed = 42
SA_data = np.load(DATA_PATH+'data_SA_updated.npy', allow_pickle=True).item()
  
properties = np.ones(95) 
masking_value = 2
SA, NSA = load_data_SA_seq(SA_data, names, offset, properties, masking_value)
 
# Merge SA nad NSA data the train and validation subsets.
all_data, all_labels = merge_data_seq(SA, NSA) 

target_len = 6
only_one_len_indices, other_len_indices = extract_len_from_data_and_labels(all_data, all_labels, target_len, masking_value)
print(len(all_labels), len(only_one_len_indices), len(other_len_indices))
train_and_validation_data_indices = other_len_indices
test_data_indices = only_one_len_indices

# Define N-fold cross validation test harness for splitting the validation from the train data
kfold_second = StratifiedKFold(n_splits=N_FOLDS_SECOND, shuffle=True, random_state=seed) 
  
test_number = -1
  
model_type = -1

# Convert train and validation indices to train and validation data and train and validation labels
train_and_validation_data, train_and_validation_labels = data_and_labels_from_indices(all_data, all_labels, train_and_validation_data_indices) 

# Convert test indices to test data and test labels
test_data, test_labels = data_and_labels_from_indices(all_data, all_labels, test_data_indices)

# Calculate weight factor for NSA peptides.
# In our data, there are more peptides that do exhibit self assembly property than are those that do not. Therefore,
# during model training, we must adjust weight factors to combat this data imbalance.
count_len_SA = 0
count_len_NSA = 0
for label in train_and_validation_labels:
    if label == 1:
        count_len_SA += 1
    else:
        count_len_NSA += 1
factor_NSA = count_len_SA / count_len_NSA 

#train_and_validation_data, test_data, train_and_validation_labels, test_labels = train_test_split(all_data, all_labels, test_size= 1 / N_FOLDS_FIRST, random_state=seed, stratify = all_labels)
 
# Write output to file
sys.stdout = open(SEQ_MODEL_DATA_PATH+str(test_number)+"_training_log_multiple_properties.txt", "w", encoding="utf-8")

# Train the ansamble model
model_training_seq(test_number, train_and_validation_data, train_and_validation_labels, kfold_second, EPOCHS, factor_NSA, test_data, test_labels, properties, names, offset, mask_value=masking_value)
 
# Close output file
sys.stdout.close()