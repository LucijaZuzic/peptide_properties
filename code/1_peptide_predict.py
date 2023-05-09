import numpy as np
from automate_training import model_predict, merge_data, load_data_SA
import sys

if len(sys.argv) > 1 and len(sys.argv[1]) <= 25: 
    dict_peptides = {sys.argv[1]: '1'}

    actual_AP = [1]

    seq_example = ''
    for i in range(24):
        seq_example += 'A'
    dict_peptides[seq_example] = '1' 

    best_batch_size = 600
    best_model = ''  
    NUM_TESTS = 5

    offset = 1 
    
    properties = np.ones(95)
    masking_value = 2

    names = ['AP']
    num_props= len(names) * 3
    SA, NSA = load_data_SA(dict_peptides, names, offset, properties, masking_value)
    all_data, all_labels = merge_data(SA, NSA) 
    print('encoded')
        
    best_model_file, best_model_image  = "../final_all/model.h5", "../final_all/model_picture.png"
    
    model_predictions_peptides = model_predict(num_props, best_batch_size, all_data, all_labels, best_model_file, best_model)[:-1]

    print(model_predictions_peptides)
else:
    if len(sys.argv) <= 1:
        print("No peptide")
    if len(sys.argv[0]) > 25:
        print("Peptide too long")