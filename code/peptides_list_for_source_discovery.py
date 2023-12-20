import numpy as np 
from utils import DATA_PATH  

# Algorithm settings 
N_FOLDS_FIRST = 5
N_FOLDS_SECOND = 5
EPOCHS = 70
names = ['AP']
offset = 1

SA_data = np.load(DATA_PATH+'data_SA_updated.npy', allow_pickle=True).item()
new_list_y = []
new_list_n = []
new_list_all = []
for x in SA_data:
    if SA_data[x] == '-1':
        continue
    if len(x) <= 24:
        if SA_data[x] == '0':
            new_list_n.append(x)
            new_list_all.append(x)
        if SA_data[x] == '1':
            new_list_y.append(x)
            new_list_all.append(x)
print(len(new_list_all))
print(len(new_list_y))
print(len(new_list_n))
new_csv = "peptide_sequence;peptide_label\n"
for peptide in new_list_all:
    new_csv += str(peptide) + ";" + str(SA_data[peptide]) + "\n"
file_csv = open(DATA_PATH+"data_for_source_discovery.csv", "w")
file_csv.write(new_csv)
file_csv.close()
new_csv = "peptide_sequence;peptide_label\n"
for peptide in new_list_y:
    new_csv += str(peptide) + ";" + str(SA_data[peptide]) + "\n"
file_csv = open(DATA_PATH+"data_for_source_discovery_yes.csv", "w")
file_csv.write(new_csv)
file_csv.close()
new_csv = "peptide_sequence;peptide_label\n"
for peptide in new_list_n:
    new_csv += str(peptide) + ";" + str(SA_data[peptide]) + "\n"
file_csv = open(DATA_PATH+"data_for_source_discovery_no.csv", "w")
file_csv.write(new_csv)
file_csv.close()