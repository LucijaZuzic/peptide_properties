from utils import DATA_PATH 
import numpy as np
import pandas as pd

df = pd.read_csv(DATA_PATH+"hasSA.csv", sep=';')
original_seq = df['pep']
original_labels = df['SA']
for i in range(len(original_labels)):
    if original_labels[i] == 'N':
        original_labels[i] = '0'
    else:
        original_labels[i] = '1'
original_dict = {}
for i in range(len(original_seq)):
    original_dict[original_seq[i]] = original_labels[i]

SA_data_old = np.load(DATA_PATH+'data_SA.npy')

peptides = []
molecules = []
dict_grades = {}
err = 0

for i in range(len(SA_data_old)):
    peptides.append(SA_data_old[i][0]) 
    errored = False
    if SA_data_old[i][0] in dict_grades and SA_data_old[i][1] != dict_grades[SA_data_old[i][0]]:
        dict_grades[SA_data_old[i][0]] = original_dict[SA_data_old[i][0]]
        err += 1
    else:   
        dict_grades[SA_data_old[i][0]] = SA_data_old[i][1]
        
print(err)

new = ['VVVVV', 
'FKFEF',
'VKVEV',
'VKVFF',
'KFFFE',
'KFAFD',
'RVSVD',
'KKFDD',
'VKVKV',
'KVKVK',
'DPDPD',
'SYCGY',
'FFEKF',
'RWLDY',
'KWEFY',
'FKIDF',
'KWMDF',
'WKPYY',
'PPPHY',
'PTPCY',
'FFEKF',
'KFFDY',
'KDHFY',
'YTEYK',
'WKPYY',
'EPYYK',
'YDPKY',
'KDPYY',
'YEPYK']

grades = [1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0]

new_tri = [
'SYY',
'YKD',
'DHK',
'PSY',
'EYK',
'TYT',
'EKW',
'EWK',
'SYH',
'YYS',
'DFK',
'DKF',
'FKD',
'KYD',
'KYF',
'KYE',
'KEY',
'WKD',
'DWK',
'KYY']

present = []
 
print(len(dict_grades))

for i in range(len(new)):
    pep = new[i]
    if pep in peptides:
        present.append(1) 
    else:
        present.append(-1)   
        dict_grades[pep] = str(grades[i])

print(len(dict_grades), len(SA_data_old))
np.save(DATA_PATH+'data_SA_updated.npy', np.array(dict_grades)) 