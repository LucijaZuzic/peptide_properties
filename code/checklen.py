# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:22:59 2022

@author: Lucija
"""


import numpy as np  
from utils import DATA_PATH  
# Algorithm settings 
N_FOLDS_FIRST = 2
N_FOLDS_SECOND = 2
EPOCHS = 30
name = 'AP'
offset = 1
# Define random seed
seed = 42
SA_data = np.load(DATA_PATH+'data_SA.npy')
suma = 0
lens = {}
len_list = []
for peptide in SA_data:
    if len(peptide[0]) > 30:
        continue
    suma += 1
    if len(peptide[0]) not in lens:
        lens[len(peptide[0])] = 1
        len_list.append(len(peptide[0]))
    else:
        lens[len(peptide[0])] += 1
 
print(len_list)
print(np.max(len_list))
print(suma, len(SA_data))
print(lens)