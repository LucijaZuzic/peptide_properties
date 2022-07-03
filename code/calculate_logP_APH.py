# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 07:05:16 2022
@author: Lucija
""" 
import numpy as np
from utils import DATA_PATH

ap_amino = np.load(DATA_PATH + 'amino_acids_AP.npy', allow_pickle=True).item()
ap_di = np.load(DATA_PATH + 'dipeptides_AP.npy', allow_pickle=True).item()
ap_tri = np.load(DATA_PATH + 'tripeptides_AP.npy', allow_pickle=True).item()

alpha = 2

logp_amino = {
    "I": -1.12,
    "L": -1.25,
    "F": -1.71,
    "V": -0.46,
    "M": -0.67,
    "P": 0.14,
    "W": -2.09,
    "H": 0.11,
    "T": 0.25,
    "Q": 0.77,
    "C": -0.02,
    "Y": -0.71,
    "A": 0.5,
    "S": 0.46,
    "N": 0.85,
    "R": 1.81,
    "G": 1.15,
    "E": 3.63,
    "K": 2.8,
    "D": 3.64,
}

logp_amino_array = [logp_amino[amino] for amino in logp_amino]
min_logp_amino = min(logp_amino_array)
max_logp_amino = max(logp_amino_array)
range_logp_amino = max_logp_amino - min_logp_amino

logp_amino_scaled = {}
for amino in logp_amino:
    logp_amino_scaled[amino] = (logp_amino[amino] - min_logp_amino) / range_logp_amino

ap_amino_scaled = {}
ap_amino_array = [ap_amino[amino] for amino in ap_amino]
min_ap_amino = min(ap_amino_array)
max_ap_amino = max(ap_amino_array)
range_ap_amino = max_ap_amino - min_ap_amino
for amino in ap_amino:
    ap_amino_scaled[amino] = (ap_amino[amino] - min_ap_amino) / range_ap_amino

aph_amino = {}
for amino in ap_amino:
    aph_amino[amino] = (ap_amino_scaled[amino] ** alpha) * logp_amino_scaled[amino]

np.save(DATA_PATH + 'amino_acids_logP.npy', np.array(logp_amino)) 

np.save(DATA_PATH + 'amino_acids_APH.npy', np.array(aph_amino)) 

logp_di = {}
for di in ap_di:
    logp = 0
    for amino in di:
        logp += logp_amino[amino]
    logp_di[di] = logp

logp_di_array = [logp_di[di] for di in logp_di]
min_logp_di = min(logp_di_array)
max_logp_di = max(logp_di_array)
range_logp_di = max_logp_di - min_logp_di

logp_di_scaled = {}
for di in logp_di:
    logp_di_scaled[di] = (logp_di[di] - min_logp_di) / range_logp_di

ap_di_scaled = {}
ap_di_array = [ap_di[di] for di in ap_di]
min_ap_di = min(ap_di_array)
max_ap_di = max(ap_di_array)
range_ap_di = max_ap_di - min_ap_di
for di in ap_di:
    ap_di_scaled[di] = (ap_di[di] - min_ap_di) / range_ap_di

aph_di = {}
for di in ap_di:
    aph_di[di] = (ap_di_scaled[di] ** alpha) * logp_di_scaled[di]

np.save(DATA_PATH + 'dipeptides_logP.npy', np.array(logp_di)) 

np.save(DATA_PATH + 'dipeptides_APH.npy', np.array(aph_di)) 

logp_tri = {}
for tri in ap_tri:
    logp = 0
    for amino in tri:
        logp += logp_amino[amino]
    logp_tri[tri] = logp

logp_tri_array = [logp_tri[tri] for tri in logp_tri]
min_logp_tri = min(logp_tri_array)
max_logp_tri = max(logp_tri_array)
range_logp_tri = max_logp_tri - min_logp_tri

logp_tri_scaled = {}
for tri in logp_tri:
    logp_tri_scaled[tri] = (logp_tri[tri] - min_logp_tri) / range_logp_tri

ap_tri_scaled = {}
ap_tri_array = [ap_tri[tri] for tri in ap_tri]
min_ap_tri = min(ap_tri_array)
max_ap_tri = max(ap_tri_array)
range_ap_tri = max_ap_tri - min_ap_tri
for tri in ap_tri:
    ap_tri_scaled[tri] = (ap_tri[tri] - min_ap_tri) / range_ap_tri

aph_tri = {}
for tri in ap_tri:
    aph_tri[tri] = (ap_tri_scaled[tri] ** alpha) * logp_tri_scaled[tri]

np.save(DATA_PATH + 'tripeptides_logP.npy', np.array(logp_tri)) 

np.save(DATA_PATH + 'tripeptides_APH.npy', np.array(aph_tri)) 

polarity_amino = {
    "I": 0,
    "L": 0,
    "F": 0,
    "V": 0,
    "M": 0,
    "P": 0,
    "W": 0,
    "H": 1,
    "T": 1,
    "Q": 1,
    "C": 0,
    "Y": 1,
    "A": 0,
    "S": 1,
    "N": 1,
    "R": 1,
    "G": 0,
    "E": 1,
    "K": 1,
    "D": 1,
}

np.save(DATA_PATH + 'amino_acids_polarity_selu.npy', np.array(polarity_amino)) 
np.save(DATA_PATH + 'amino_acids_polarity_relu.npy', np.array(polarity_amino)) 

polarity_di = {}
for di in ap_di:
    polarity = 0
    for amino in di:
        polarity += polarity_amino[amino]
    polarity_di[di] = polarity

np.save(DATA_PATH + 'dipeptides_polarity_selu.npy', np.array(polarity_di)) 
np.save(DATA_PATH + 'dipeptides_polarity_relu.npy', np.array(polarity_di)) 

polarity_tri = {}
for tri in ap_tri:
    polarity = 0
    for amino in tri:
        polarity += polarity_amino[amino]
    polarity_tri[tri] = polarity

np.save(DATA_PATH + 'tripeptides_polarity_selu.npy', np.array(polarity_tri)) 
np.save(DATA_PATH + 'tripeptides_polarity_relu.npy', np.array(polarity_tri))  