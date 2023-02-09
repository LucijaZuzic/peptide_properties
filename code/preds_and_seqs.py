from utils import PATH_TO_NAME, predictions_name, DATA_PATH, SEQ_MODEL_DATA_PATH, MODEL_DATA_PATH, MY_MODEL_DATA_PATH, setSeed, PATH_TO_EXTENSION
from custom_plots import results_name 
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

MAXLEN = 24
def read_one_prediction(some_path, test_number, final_model_type, iteration, some_seed):
    file = open(predictions_name(some_path, test_number, final_model_type, iteration), "r")
    lines = file.readlines()
    predictions = eval(lines[0])
    labels = eval(lines[1])
    file.close()
    file_df = pd.read_csv("../seeds/seed_" + str(some_seed) + "/similarity/" + 'test_fold_' + str(test_number) + '.csv')
    other_lables = file_df['label']
    sequences = file_df['sequence']  

    return predictions, labels, sequences

def read_all_model_predictions(some_path, min_test_number, max_test_number, final_model_type, iteration, some_seed):
    all_predictions = []
    all_labels = []
    all_sequences = []
    for test_number in range(min_test_number, max_test_number + 1): 
        predictions, labels, sequences = read_one_prediction(some_path, test_number, final_model_type, iteration, some_seed)
        for prediction in predictions:
            all_predictions.append(prediction)
        for label in labels:
            all_labels.append(label) 
        for sequence in sequences:
            all_sequences.append(sequence) 
    return all_predictions, all_labels, all_sequences 

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
paths = [SEQ_MODEL_DATA_PATH, MODEL_DATA_PATH, MY_MODEL_DATA_PATH] 
NUM_TESTS = 5 

dict_some = {} 

aa_preds = [] 
aa_labels_new= []
aa_model_types = []

for some_path in paths:

    all_preds = []
    all_seqs = []
    all_labels_new= []
    all_model_types = []
    all_seeds = []

    for seed in seed_list: 
        setSeed(seed)
        all_predictions, all_labels, all_sequences = read_all_model_predictions(some_path, 1, 5, "weak", 1, seed)
        for pred in all_predictions: 
            all_preds.append(pred)
            aa_preds.append(pred)
            all_seeds.append(seed)
        for sequence in all_sequences: 
            all_seqs.append(sequence) 
        for label in all_labels: 
            if label == 1:
                all_labels_new.append('SA') 
                aa_labels_new.append('SA') 
            else:
                all_labels_new.append('NSA') 
                aa_labels_new.append('NSA') 
            all_model_types.append(PATH_TO_NAME[some_path].replace("SP and AP", "Hybrid AP-SP"))
            aa_model_types.append(PATH_TO_NAME[some_path].replace("SP and AP", "Hybrid AP-SP"))

    dict_some[PATH_TO_NAME[some_path].replace("SP and AP", "Hybrid AP-SP")] = {'sequences': all_seqs, 'labels': all_labels_new, 'predictions': all_preds, 'seeds': all_seeds, 'model_types': all_model_types}

dictionary_final = {'sequences': dict_some['AP']['sequences'], 
                    'labels': dict_some['AP']['labels'], 
                    'seeds': dict_some['AP']['seeds'], 
                    'AP': dict_some['AP']['predictions'], 
                    'SP': dict_some['SP']['predictions'], 
                    'AP-SP': dict_some['Hybrid AP-SP']['predictions']} 
  
header = "sequences;labels;seeds;AP;SP;AP-SP;\n"
write_all = ""

for i in range(5):
    write_parts = {}  
    for j in range(368*i,368*(i+1)): 
        write_parts[dictionary_final['sequences'][j]] = dictionary_final['sequences'][j] + ";" + str(dictionary_final['labels'][j]) + ";" + str(dictionary_final['seeds'][j]) + ";" + str(dictionary_final['AP'][j]) + ";" + str(dictionary_final['SP'][j]) + ";" + str(dictionary_final['AP-SP'][j]) + "\n"
    write_str = ""
    print(sorted(write_parts))
    for str_write in sorted(write_parts):
        write_str += write_parts[str_write] 
    write_all += write_str 
    file_output = open("../data/sequence_pred_seed_" + str(seed_list[i] )+ ".csv", "w", encoding="utf-8") 
    file_output.write(header + write_str.replace('.', ','))
    file_output.close()

file_output = open("../data/sequence_pred_all.csv", "w", encoding="utf-8") 
file_output.write(header + write_all.replace('.', ','))
file_output.close()


plt.rcParams.update({'font.size': 22})
d = {'Predicted self assembly probability': aa_preds, 'Self assembly status': aa_labels_new, 'Model': aa_model_types}
df = pd.DataFrame(data=d)
plt.figure()
g = sns.displot(data=df, x = 'Predicted self assembly probability', kde=True, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], hue = 'Self assembly status', col = 'Model', palette = {'NSA': '#ff120a', 'SA': '#2e85ff'})
g.set_axis_labels("Self assembly probability", "Number of peptides")
g.set_titles("{col_name} model")  
plt.show()
plt.close() 