# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 05:58:17 2022

@author: Lucija
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from seqprops import SequentialPropertiesEncoder 

# Some input data
sequences = ["AAC", "ACACA", "AHHHTK", "HH"]
y = np.array([0, 1, 1, 0])

# Encode sequences
encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-1, 1)))
X = encoder.encode(sequences)
 
print(X)