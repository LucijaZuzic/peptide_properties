from rdkit import Chem
from mordred import Calculator, descriptors
from utils import DATA_PATH
import numpy as np

# create descriptor calculator with all descriptors
calc = Calculator(descriptors, ignore_3D=True)

len(calc.descriptors)
len(Calculator(descriptors, ignore_3D=True, version="1.0.0"))

SA_data = np.load(DATA_PATH+'data_SA.npy')

for peptide in SA_data:
    if len(peptide[0]) > 24:
        continue
    mol = Chem.MolFromSequence(peptide[0])
    print(calc(mol)[:3])    
    
# calculate single molecule
mol = Chem.MolFromSmiles('c1ccccc1')
print(calc(mol)[:3])

# calculate multiple molecule
#mols = [Chem.MolFromSmiles(smi) for smi in ['c1ccccc1Cl', 'c1ccccc1O', 'c1ccccc1N']]

# as pandas
#df = calc.pandas(mols)
#df['SLogP']