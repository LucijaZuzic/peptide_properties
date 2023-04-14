import pandas as pd
import numpy as np
from utils import DATA_PATH, SEQ_MODEL_DATA_PATH, MODEL_DATA_PATH, MY_MODEL_DATA_PATH, predictions_name, setSeed, final_h5_and_png
from automate_training import common_no_file_after_training, merge_data_AP, merge_data_seq, merge_data, model_predict, model_predict_AP, model_predict_seq, load_data_SA_AP, load_data_SA_seq, load_data_SA
df = pd.read_csv(DATA_PATH + "human_AI.csv", sep = ";")

def human_AI_predictions_name(some_path, number, final_model_type, iteration):
    return predictions_name(some_path, number, final_model_type, iteration).replace("predictions", "human_AI_predictions")
  
dict_human_AI = {}
for i in range(len(df['pep'])):
    dict_human_AI[df['pep'][i]] = str(df['agg'][i])
 
seq_example = ''
for i in range(24):
    seq_example += 'A'
dict_human_AI[seq_example] = '1' 

best_batch_size = 600
best_model = '' 
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276] 
NUM_TESTS = 5

offset = 1 
  
properties = np.ones(95)
masking_value = 2
'''
for seed in seed_list:
    setSeed(seed)
    print(seed)
    for number in range(1, NUM_TESTS + 1):
        print(number)
        best_model_file, best_model_image = final_h5_and_png(MY_MODEL_DATA_PATH, number, 1)
        common_no_file_after_training(0.5, number, MY_MODEL_DATA_PATH, 'weak', 1, properties, best_model_file, '', names=['AP'], offset = 1, masking_value=2)
 
        best_model_file, best_model_image = final_h5_and_png(MODEL_DATA_PATH, number, 1)
        common_no_file_after_training(0.5, number, MODEL_DATA_PATH, 'weak', 1, properties, best_model_file, '', names=['AP'], offset = 1, masking_value=2)
    
        best_model_file, best_model_image = final_h5_and_png(SEQ_MODEL_DATA_PATH, number, 1)
        common_no_file_after_training(0.5, number, SEQ_MODEL_DATA_PATH, 'weak', 1, properties, best_model_file, '', names= ['AP', 'logP', 'APH', 'polarity_selu'], offset = 1, masking_value=2)
'''
names = ['AP']
num_props= len(names) * 3
SA_AP, NSA_AP = load_data_SA_AP(dict_human_AI, names, offset, properties, masking_value)
all_data_AP, all_labels_AP = merge_data_AP(SA_AP, NSA_AP) 
print('encoded AP')

names = []
num_props= len(names) * 3
SA_SEQ, NSA_SEQ = load_data_SA_seq(dict_human_AI, names, offset, properties, masking_value)
all_data_SEQ, all_labels_SEQ = merge_data_seq(SA_SEQ, NSA_SEQ) 
print('encoded SEQ')

names = ['AP']
num_props= len(names) * 3
SA, NSA = load_data_SA(dict_human_AI, names, offset, properties, masking_value)
all_data, all_labels = merge_data(SA, NSA) 
print('encoded')

for seed in seed_list:
    setSeed(seed)
    print(seed)
    for number in range(1, NUM_TESTS + 1): 

        print(number)
        
        best_model_file, best_model_image = final_h5_and_png(SEQ_MODEL_DATA_PATH, number, 1)
        common_no_file_after_training(0.5, number, SEQ_MODEL_DATA_PATH, 'weak', 1, properties, best_model_file, '', names= [], offset = 1, masking_value=2)

        model_predictions_seq_human_AI = model_predict_seq(best_batch_size, all_data_SEQ, all_labels_SEQ, best_model_file, best_model)

        other_output = open(
            human_AI_predictions_name(SEQ_MODEL_DATA_PATH, number, 'weak', 1),
            "w",
            encoding="utf-8",
        )
        other_output.write(str(model_predictions_seq_human_AI))
        other_output.write("\n") 
        other_output.close()
        print(human_AI_predictions_name(SEQ_MODEL_DATA_PATH, number, 'weak', 1))

        best_model_file, best_model_image = final_h5_and_png(MY_MODEL_DATA_PATH, number, 1)
        common_no_file_after_training(0.5, number, MY_MODEL_DATA_PATH, 'weak', 1, properties, best_model_file, '', names=['AP'], offset = 1, masking_value=2)

        model_predictions_AP_human_AI = model_predict_AP(num_props, best_batch_size, all_data_AP, all_labels_AP, best_model_file, best_model)

        other_output = open(
            human_AI_predictions_name(MY_MODEL_DATA_PATH, number, 'weak', 1),
            "w",
            encoding="utf-8",
        )
        other_output.write(str(model_predictions_AP_human_AI))
        other_output.write("\n") 
        other_output.close()
        print(human_AI_predictions_name(MY_MODEL_DATA_PATH, number, 'weak', 1))

        best_model_file, best_model_image = final_h5_and_png(MODEL_DATA_PATH, number, 1)
        common_no_file_after_training(0.5, number, MODEL_DATA_PATH, 'weak', 1, properties, best_model_file, '', names=['AP'], offset = 1, masking_value=2)

        model_predictions_human_AI = model_predict(num_props, best_batch_size, all_data, all_labels, best_model_file, best_model)

        other_output = open(
            human_AI_predictions_name(MODEL_DATA_PATH, number, 'weak', 1),
            "w",
            encoding="utf-8",
        )
        other_output.write(str(model_predictions_human_AI))
        other_output.write("\n") 
        other_output.close()
        print(human_AI_predictions_name(MODEL_DATA_PATH, number, 'weak', 1)) 