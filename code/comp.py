print(2 == 2.0)
from seqprops import SequentialPropertiesEncoder 
from sklearn.preprocessing import MinMaxScaler 
import numpy as np
from utils import DATA_PATH
from automate_training import reshape, reshape_AP, reshape_seq, load_data_SA_seq, load_data_SA, load_data_SA_AP, load_data_AP, split_amino_acids, split_dipeptides, split_tripeptides, padding
seq = ''
index = 0
seqs = []
SA_data = np.load(DATA_PATH+'data_SA_updated.npy', allow_pickle=True).item()
maxlen = 24
maxilen = 0
YES = []
NO = []
for i in range(len(SA_data)):
    x = list(SA_data.keys())[i]
    maxilen = max(maxilen, len(x))
    if len(x) > maxlen:
        continue
    if SA_data[x] == '-1':
        continue
    if SA_data[x] == '1':
        YES.append(x)
    if SA_data[x] == '0':
        NO.append(x)
    seqs.append(x)
    if len(x) == 6 and SA_data[x] == '1':
        rep = False
        for l in x: 
            if x.count(l) > 1:
                rep = True
                break
        if rep:
            continue
        else:
            if seq == '':
                print(x, SA_data[x]) 
                seq = x
                index = i

print(seqs[index])     
offset = 1
masking_value = 2
properties = np.ones(95)
encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-1, 1)))
encoded_sequences = encoder.encode(seqs)   
print(len(seqs))  
print(len(encoded_sequences))
print(len(encoded_sequences[0]))
print(len(encoded_sequences[0][0]))
print(seqs[index]) 

masking_value = 2
offset = 1
header = 'AminoAcid,Hydrophobicity_Aboderin,Hydrophobicity_AbrahamLeo,Hydrophobicity_Argos,Hydrophobicity_BlackMould,Hydrophobicity_BullBreese,Hydrophobicity_Casari,Hydrophobicity_Chothia,Hydrophobicity_Cid,Hydrophobicity_Cowan3.4,Hydrophobicity_Cowan7.5,Hydrophobicity_Eisenberg,Hydrophobicity_Engelman,Hydrophobicity_Fasman,Hydrophobicity_Fauchere,Hydrophobicity_Goldsack,Hydrophobicity_Guy,Hydrophobicity_HoppWoods,Hydrophobicity_Janin,Hydrophobicity_Jones,Hydrophobicity_Juretic,Hydrophobicity_Kidera,Hydrophobicity_Kuhn,Hydrophobicity_KyteDoolittle,Hydrophobicity_Levitt,Hydrophobicity_Manavalan,Hydrophobicity_Miyazawa,Hydrophobicity_Parker,Hydrophobicity_Ponnuswamy,Hydrophobicity_Prabhakaran,Hydrophobicity_Rao,Hydrophobicity_Rose,Hydrophobicity_Roseman,Hydrophobicity_Sweet,Hydrophobicity_Tanford,Hydrophobicity_Welling,Hydrophobicity_Wilson,Hydrophobicity_Wolfenden,Hydrophobicity_Zimmerman,crucianiProperties_PP1,crucianiProperties_PP2,crucianiProperties_PP3,zScales_Z1,zScales_Z2,zScales_Z3,zScales_Z4,zScales_Z5,FASGAI_F1,FASGAI_F2,FASGAI_F3,FASGAI_F4,FASGAI_F5,FASGAI_F6,VHSE_VHSE1,VHSE_VHSE2,VHSE_VHSE3,VHSE_VHSE4,VHSE_VHSE5,VHSE_VHSE6,VHSE_VHSE7,VHSE_VHSE8,ProtFP_ProtFP1,ProtFP_ProtFP2,ProtFP_ProtFP3,ProtFP_ProtFP4,ProtFP_ProtFP5,ProtFP_ProtFP6,ProtFP_ProtFP7,ProtFP_ProtFP8,BLOSUM_BLOSUM1,BLOSUM_BLOSUM2,BLOSUM_BLOSUM3,BLOSUM_BLOSUM4,BLOSUM_BLOSUM5,BLOSUM_BLOSUM6,BLOSUM_BLOSUM7,BLOSUM_BLOSUM8,BLOSUM_BLOSUM9,BLOSUM_BLOSUM10,MSWHIM_MSWHIM1,MSWHIM_MSWHIM2,MSWHIM_MSWHIM3,tScales_T1,tScales_T2,tScales_T3,tScales_T4,tScales_T5,stScales_ST1,stScales_ST2,stScales_ST3,stScales_ST4,stScales_ST5,stScales_ST6,stScales_ST7,stScales_ST8'
header = header.split(',')

vals = ['Hydrophobicity_Aboderin', 'FASGAI_F3', 'VHSE_VHSE1']
indexes = {}
for i in range(len(header)):
    if header[i] in vals:
        print(i, header[i]) 
        indexes[header[i]] = i
print(indexes)
for i in range(index, index + 1):
    print(seqs[i]) 
    for j in range(0, len(seqs[i])):
        for k in indexes:
            print(encoded_sequences[i][j][indexes[k]])
        print(" ")
        
    for name in ['AP', 'logP', 'APH', 'polarity_selu']:
        amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP(name, offset)  
        amino_acids_ap = split_amino_acids(seqs[index], amino_acids_AP)
        dipeptides_ap = split_dipeptides(seqs[index], dipeptides_AP)
        tripeptides_ap = split_tripeptides(seqs[index], tripeptides_AP)
                
        amino_acids_ap_padded = padding(amino_acids_ap, len(encoded_sequences[index]), masking_value)
        dipeptides_acids_ap_padded = padding(dipeptides_ap, len(encoded_sequences[index]), masking_value)
        tripeptides_ap_padded = padding(tripeptides_ap, len(encoded_sequences[index]), masking_value)  
 
        print(amino_acids_ap_padded)
        print(dipeptides_acids_ap_padded)
        print(tripeptides_ap_padded)
 
resulteval = DATA_PATH+'RESULTEVAL.csv' 

import pandas as pd
df = pd.read_csv(resulteval, skipinitialspace=True, sep=';')
sequences = list(df['Dizajnirani peptid'])
seq_example = ''
for i in range(24):
    seq_example += 'A'
sequences.append(seq_example)
past_grades = list(df['Postotak svojstava koja imaju AUC > 0,5 koja su dala SA'])
past_classes= list(df['SA']) 

SA_data = {}
for i in range(len(sequences)):
    SA_data[sequences[i]] = '0'
 
properties = np.ones(95)
names=['AP', 'logP', 'APH', 'polarity_selu']
SAs, NSAs = load_data_SA_seq(SA_data, names, offset, properties, masking_value) 
names=['AP']
SAa, NSAa = load_data_SA_AP(SA_data, names, offset, properties, masking_value) 
SA, NSA = load_data_SA(SA_data, names, offset, properties, masking_value) 
  
SA_data = np.load(DATA_PATH+'data_SA_updated.npy', allow_pickle=True).item()
 
properties = np.ones(95)
names=['AP', 'logP', 'APH', 'polarity_selu']
SAs, NSAs = load_data_SA_seq(SA_data, names, offset, properties, masking_value) 
names=['AP']
SAa, NSAa = load_data_SA_AP(SA_data, names, offset, properties, masking_value) 
SA, NSA = load_data_SA(SA_data, names, offset, properties, masking_value) 
    
print(len(SAs), len(NSAs))
print(len(SAa), len(NSAa))
print(len(SA), len(NSA))

print(len(SAs[0]), len(NSAs[0]))
print(len(SAa[0]), len(NSAa[0]))
print(len(SA[0]), len(NSA[0]))

print(len(SAs[0][0]), len(NSAs[0][0]))
print(len(SAa[0][0]), len(NSAa[0][0]))
print(len(SA[0][0]), len(NSA[0][0]))
SA2 = []
for i in range(0, len(SAs)): 
    SA2.append(np.array(SAs[i]).transpose())
NSA2 = []
for i in range(0, len(NSAs)): 
    NSA2.append(np.array(NSAs[i]).transpose())
index = 12
print(YES[index]) 
for i in range(index, index + 1):  
    for j in range(len(SA2[i]) - 2, len(SA2[i])): 
        print(SA2[i][j])
for i in range(index, index + 1):  
    for j in range(len(SAa[i]) - 2, len(SAa[i])): 
        print(SAa[i][j])
for i in range(index, index + 1):  
    for j in range(len(SA[i]) - 2, len(SA[i])): 
        print(SA[i][j])
 
 
print(len(header))
print(len(SA[0])) 
print(len(SA[0][0]))