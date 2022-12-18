import numpy as np 
from utils import DATA_PATH 
from sklearn.model_selection import StratifiedKFold
from plot_similarities_seqprops import main_sim, double_sim

# Define random seed
seed = 42
# Algorithm settings 
N_FOLDS_FIRST = 5
SA_data = np.load(DATA_PATH+'data_SA_updated.npy', allow_pickle=True).item()
sequences = []
labels = []
MAXLEN = 24
sequences_6 = []
labels_6 = []
sequences_not_6 = []
labels_not_6 = []

for peptide in SA_data:
    if len(peptide) > MAXLEN or SA_data[peptide] == '-1':
        continue
    sequences.append(peptide)
    labels.append(SA_data[peptide])
    if len(peptide) == 6:
        sequences_6.append(peptide)
        labels_6.append(SA_data[peptide])
    else:
        sequences_not_6.append(peptide)
        labels_not_6.append(SA_data[peptide])

# Define N-fold cross validation test harness for splitting the test data from the train and validation data
kfold_first = StratifiedKFold(n_splits=N_FOLDS_FIRST, shuffle=True, random_state=seed)

test_number = 0
for train_and_validation_data_indices, test_data_indices in kfold_first.split(sequences, labels):
    test_number += 1

    # Convert train and validation indices to train and validation data and train and validation labels
    train_test_save = "sequence,label\n"
    train_save = "sequence,label\n" 
    for i in train_and_validation_data_indices: 
        train_save += sequences[i] + "," + labels[i] + "\n" 
        train_test_save += sequences[i] + "," + labels[i] + "\n"

    train_output = open(DATA_PATH + 'train_fold_' + str(test_number) + ".csv", "w", encoding="utf-8") 
    train_output.write(train_save)
    train_output.close()
    main_sim('train_fold_' + str(test_number))
         
    # Convert test indices to test data and test labels
    test_save = "sequence,label\n"
    test_data = []
    test_labels = [] 
    for i in test_data_indices: 
        test_save += sequences[i] + "," + labels[i] + "\n" 
        train_test_save += sequences[i] + "," + labels[i] + "\n" 

    test_output = open(DATA_PATH + 'test_fold_' + str(test_number) + ".csv", "w", encoding="utf-8") 
    test_output.write(test_save)
    test_output.close()
    main_sim('test_fold_' + str(test_number))

    train_test_output = open(DATA_PATH + 'train_test_fold_' + str(test_number) + ".csv", "w", encoding="utf-8") 
    train_test_output.write(train_test_save)
    train_test_output.close()
    double_sim('train_test_fold_' + str(test_number), len(train_and_validation_data_indices))
 
all_save = "sequence,label\n"
only_6_save = "sequence,label\n"
only_6_data = []
only_6_labels = [] 
for i in range(len(sequences_6)):
    only_6_data.append(sequences_6[i])
    only_6_labels.append(labels_6[i]) 
    only_6_save += only_6_data[-1] + "," + only_6_labels[-1] + "\n" 
    all_save += only_6_data[-1] + "," + only_6_labels[-1] + "\n" 

only_6_output = open(DATA_PATH + 'only_6.csv', "w", encoding="utf-8") 
only_6_output.write(only_6_save)
only_6_output.close()
main_sim('only_6')
        
only_not_6_save = "sequence,label\n"
only_not_6_data = []
only_not_6_labels = [] 
for i in range(len(sequences_not_6)):
    only_not_6_data.append(sequences_not_6[i])
    only_not_6_labels.append(labels_not_6[i]) 
    only_not_6_save += only_not_6_data[-1] + "," + only_not_6_labels[-1] + "\n" 
    all_save += only_not_6_data[-1] + "," + only_not_6_labels[-1] + "\n" 

only_not_6_output = open(DATA_PATH + 'only_not_6.csv', "w", encoding="utf-8") 
only_not_6_output.write(only_not_6_save)
only_not_6_output.close()
main_sim('only_not_6')

all_output = open(DATA_PATH + "all_similarity.csv", "w", encoding="utf-8") 
all_output.write(all_save)
all_output.close()
main_sim('all_similarity')

divide_by_6_output = open(DATA_PATH + "divide_by_6.csv", "w", encoding="utf-8") 
divide_by_6_output.write(all_save)
divide_by_6_output.close()
double_sim('divide_by_6', len(only_6_data))