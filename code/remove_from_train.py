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

df2 = pd.read_csv(DATA_PATH + "human_AI.csv", sep = ";")

banned = ['IHIHIQI', 'IHIHINI', 'IHINIHI']

for x2 in df2['pep']:
    banned.append(x2)

#print(df2['pep'])
dict_grades2 = {}
for x2 in dict_grades:
    #print("yes", x2)
    if x2 in banned:
        #print("skip", x2)
        continue
    dict_grades2[x2] = dict_grades[x2]

print(len(df2['pep']), len(dict_grades2))
np.save(DATA_PATH+'data_SA_no_updated.npy', np.array(dict_grades2))

banned_no_AI = ['IHIHIQI', 'IHIHINI', 'IHINIHI']

for i in range(len(df2['pep'])):
    if df2['expert'][i] != "AI":
        continue
    banned_no_AI.append(df2['pep'][i])

#print(df2['pep'])
dict_grades2_no_AI = {}
for x2 in dict_grades:
    #print("yes", x2)
    if x2 in banned_no_AI:
        #print("skip", x2)
        continue
    dict_grades2_no_AI[x2] = dict_grades[x2]

print(len(df2['pep']), len(banned_no_AI), len(dict_grades2_no_AI))
np.save(DATA_PATH+'data_SA_no_AI_updated.npy', np.array(dict_grades2_no_AI))

banned_no_human = ['IHIHIQI', 'IHIHINI', 'IHINIHI']

for i in range(len(df2['pep'])):
    if df2['expert'][i] == "AI":
        continue
    banned_no_human.append(df2['pep'][i])

#print(df2['pep'])
dict_grades2_no_human = {}
for x2 in dict_grades:
    #print("yes", x2)
    if x2 in banned_no_human:
        #print("skip", x2)
        continue
    dict_grades2_no_human[x2] = dict_grades[x2]

print(len(df2['pep']), len(banned_no_human), len(dict_grades2_no_human))
np.save(DATA_PATH+'data_SA_no_human_updated.npy', np.array(dict_grades2_no_human))
