######################################################################################################
##  This script plots the distribution of similarities among all peptides from a specified dataset  ##
######################################################################################################

import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from skbio import Protein
from skbio.alignment import global_pairwise_align_protein
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from utils import DATA_PATH

# Set font size for plot
plt.rcParams.update({'font.size': 14})
figure(figsize=(8, 6), dpi=80)

# The function that computes similarity for two sequences.
def compute_similarity(seq1, seq2):
    aln, _, _  = global_pairwise_align_protein(
                Protein(seq1),
                Protein(seq2)
            )    
    seq1, seq2 = aln
    return seq1.match_frequency(seq2, relative=True)

def main_sim(read_from, save_to):
    # Load sequences from a CSV file
    data = pd.read_csv(read_from)
    sequences = data["sequence"].to_numpy()

    tasks = []
    for idx, seq1 in enumerate(sequences):
        for seq2 in sequences[idx + 1:]:
            tasks.append((seq1, seq2))
    
    with Pool() as p:
        similarities = p.starmap(compute_similarity, tasks)

    plt.hist(similarities, bins=100, weights=np.ones_like(similarities) / len(similarities))

    plt.ylabel("Frequency")
    plt.xlabel("Similarity") 
    plt.savefig(save_to) 
    plt.close()

def double_sim(read_from, save_to, divider):
    # Load sequences from a CSV file
    data = pd.read_csv(read_from)
    sequences = data["sequence"].to_numpy()

    tasks = []
    for seq1 in sequences[:divider]:
        for seq2 in sequences[divider:]:
            tasks.append((seq1, seq2))
    
    with Pool() as p:
        similarities = p.starmap(compute_similarity, tasks)

    plt.hist(similarities, bins=100, weights=np.ones_like(similarities) / len(similarities))

    plt.ylabel("Frequency")
    plt.xlabel("Similarity") 
    plt.savefig(save_to) 
    plt.close()