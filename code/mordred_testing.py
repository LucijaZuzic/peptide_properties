from rdkit import Chem
from mordred import Calculator, descriptors
from utils import DATA_PATH, padding
import numpy as np
import pandas as pd
import numbers

# create descriptor calculator with all descriptors
calc = Calculator(descriptors, ignore_3D=True)

len(calc.descriptors)
len(Calculator(descriptors, ignore_3D=True, version="1.0.0"))

SA_data = np.load(DATA_PATH+'data_SA.npy')

peptides = []
molecules = []
for peptide in SA_data[:, 0]:
    if len(peptide) > 24:
        continue
    peptides.append(peptide)
    molecules.append(Chem.MolFromSequence(peptide))

df = calc.pandas(molecules)

na_cols = []
non_na_cols = []
for col in df.columns:
    is_valid = True
    for sth in df[col]:
        if isinstance(str, numbers.Number) == False:
            na_cols.append(col)
            is_valid = False
            break
    if is_valid:
        non_na_cols.append(col)

print(len(df.columns), len(na_cols), len(non_na_cols))
df['sequence'] = peptides
df.to_csv(DATA_PATH+"mordred_descriptors.csv", sep=';')