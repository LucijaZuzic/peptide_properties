import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy import stats
from utils import DATA_PATH
from automate_training import merge_data_AP, merge_data_seq, merge_data, model_predict, model_predict_AP, model_predict_seq, load_data_SA_AP, load_data_SA_seq, load_data_SA
df = pd.read_csv(DATA_PATH + "41557_2022_1055_MOESM3_ESM_Figure3a_5mer_score_shortMD.csv")
plt.rcParams.update({'font.size': 22})
def myfunc(x):
  return slope * x + intercept

dict_hex = {}
for i in df['pep']:
    dict_hex[i] = '1' 

actual_AP = []
for i in df['AP']:
    actual_AP.append(i) 
'''
seq_example = ''
for i in range(24):
    seq_example += 'A'
dict_hex[seq_example] = '1' 

best_batch_size = 600
best_model = ''  
NUM_TESTS = 5

offset = 1 
  
properties = np.ones(95)
masking_value = 2

names = ['AP']
num_props= len(names) * 3
SA_AP, NSA_AP = load_data_SA_AP(dict_hex, names, offset, properties, masking_value)
all_data_AP, all_labels_AP = merge_data_AP(SA_AP, NSA_AP) 
print('encoded AP')

names = []
num_props= len(names) * 3
SA_SEQ, NSA_SEQ = load_data_SA_seq(dict_hex, names, offset, properties, masking_value)
all_data_SEQ, all_labels_SEQ = merge_data_seq(SA_SEQ, NSA_SEQ) 
print('encoded SEQ')

names = ['AP']
num_props= len(names) * 3
SA, NSA = load_data_SA(dict_hex, names, offset, properties, masking_value)
all_data, all_labels = merge_data(SA, NSA) 
print('encoded')
    
best_model_file, best_model_image  = "../final_seq/model.h5", "../final_seq/model_picture.png"

model_predictions_seq_hex = model_predict_seq(best_batch_size, all_data_SEQ, all_labels_SEQ, best_model_file, best_model)[:-1]

other_output = open(
    "../final_seq/hex_predict.txt",
    "w",
    encoding="utf-8",
)
other_output.write(str(model_predictions_seq_hex))
other_output.write("\n") 
other_output.close() 

plt.figure()
plt.title(
    "Model SP"
)
plt.xlabel("Predicted self assembly probability")
plt.ylabel("AP")
plt.scatter(
    model_predictions_seq_hex,
    actual_AP
) 
slope, intercept, r, p, std_err = stats.linregress(model_predictions_seq_hex, actual_AP)
print("Seq. model R: " + str(r)) 
print("Seq. model corrcoef: " + str(np.corrcoef(model_predictions_seq_hex, actual_AP)[0][1])) 
print("Seq. model spearmanr: " + str(stats.spearmanr(model_predictions_seq_hex, actual_AP)[0])) 
print("Seq. model R2: " + str(sklearn.metrics.r2_score(model_predictions_seq_hex, actual_AP)))  
mymodel = list(map(myfunc, model_predictions_seq_hex)) 
plt.plot(
    model_predictions_seq_hex,
    mymodel, color = 'r'
)  
plt.savefig("../final_seq/hex_predict.png", bbox_inches="tight")
plt.close() 

best_model_file, best_model_image  = "../final_AP/model.h5", "../final_AP/model_picture.png" 

model_predictions_AP_hex = model_predict_AP(num_props, best_batch_size, all_data_AP, all_labels_AP, best_model_file, best_model)[:-1]

other_output = open(
    "../final_AP/hex_predict.txt",
    "w",
    encoding="utf-8",
)
other_output.write(str(model_predictions_AP_hex))
other_output.write("\n") 
other_output.close() 

plt.figure()
plt.title(
    "Model AP"
)
plt.xlabel("Predicted self assembly probability")
plt.ylabel("AP")
plt.scatter(
    model_predictions_AP_hex,
    actual_AP
)  
slope, intercept, r, p, std_err = stats.linregress(model_predictions_AP_hex, actual_AP)
print("AP model R: " + str(r)) 
print("AP model corrcoef: " + str(np.corrcoef(model_predictions_AP_hex, actual_AP)[0][1])) 
print("AP model spearmanr: " + str(stats.spearmanr(model_predictions_AP_hex, actual_AP)[0])) 
print("AP model R2: " + str(sklearn.metrics.r2_score(model_predictions_AP_hex, actual_AP)))  
mymodel = list(map(myfunc, model_predictions_AP_hex)) 
plt.plot(
    model_predictions_AP_hex,
    mymodel, color = 'r'
)  
plt.savefig("../final_AP/hex_predict.png", bbox_inches="tight")
plt.close() 

best_model_file, best_model_image  = "../final_all/model.h5", "../final_all/model_picture.png" 

model_predictions_hex = model_predict(num_props, best_batch_size, all_data, all_labels, best_model_file, best_model)[:-1]

other_output = open(
    "../final_all/hex_predict.txt",
    "w",
    encoding="utf-8",
)
other_output.write(str(model_predictions_hex))
other_output.write("\n") 
other_output.close()

plt.figure()
plt.title(
    "Model SP and AP"
)
plt.xlabel("Predicted self assembly probability")
plt.ylabel("AP")
plt.scatter(
    model_predictions_hex,
    actual_AP
)  
slope, intercept, r, p, std_err = stats.linregress(model_predictions_hex, actual_AP)
print("Seq. and AP model R: " + str(r)) 
print("Seq. and AP model corrcoef: " + str(np.corrcoef(model_predictions_hex, actual_AP)[0][1])) 
print("Seq. and AP model spearmanr: " + str(stats.spearmanr(model_predictions_hex, actual_AP)[0])) 
print("Seq. and AP model R2: " + str(sklearn.metrics.r2_score(model_predictions_hex, actual_AP)))   
mymodel = list(map(myfunc, model_predictions_hex)) 
plt.plot(
    model_predictions_hex,
    mymodel, color = 'r'
)  
plt.savefig("../final_all/hex_predict.png", bbox_inches="tight")
plt.close() 
'''

other_output = open(
    "../final_all/hex_predict.txt",
    "r",
    encoding="utf-8",
) 
model_predictions_hex = eval(other_output.readlines()[0])
other_output.close()

other_output = open(
    "../final_AP/hex_predict.txt",
    "r",
    encoding="utf-8",
)
model_predictions_AP_hex = eval(other_output.readlines()[0])
other_output.close()

other_output = open(
    "../final_seq/hex_predict.txt",
    "r",
    encoding="utf-8",
)
model_predictions_seq_hex = eval(other_output.readlines()[0])
other_output.close()

plt.figure(figsize=(25, 5))
plt.subplot(1, 3, 1)
plt.title(
    "Model SP"
)
plt.xlabel("Predicted self assembly probability")
plt.ylabel("AP")
plt.xticks([0.25, 0.5, 0.75])
plt.scatter(
    model_predictions_seq_hex,
    actual_AP
) 
slope, intercept, r, p, std_err = stats.linregress(model_predictions_seq_hex, actual_AP)
mymodel = list(map(myfunc, model_predictions_seq_hex)) 
plt.plot(
    model_predictions_seq_hex,
    mymodel, color = 'r'
)  
plt.subplot(1, 3, 2)
plt.title(
    "Model SP and AP"
)
plt.xlabel("Predicted self assembly probability")
plt.yticks([])
plt.xticks([0.25, 0.5, 0.75])
plt.scatter(
    model_predictions_hex,
    actual_AP
) 
slope, intercept, r, p, std_err = stats.linregress(model_predictions_hex, actual_AP)
mymodel = list(map(myfunc, model_predictions_hex)) 
plt.plot(
    model_predictions_hex,
    mymodel, color = 'r'
)  
plt.subplot(1, 3, 3)
plt.title(
    "Model AP"
)
plt.xlabel("Predicted self assembly probability")
plt.yticks([])
plt.xticks([0.25, 0.5, 0.75])
plt.scatter(
    model_predictions_AP_hex,
    actual_AP
) 
slope, intercept, r, p, std_err = stats.linregress(model_predictions_AP_hex, actual_AP)
mymodel = list(map(myfunc, model_predictions_AP_hex)) 
plt.plot(
    model_predictions_AP_hex,
    mymodel, color = 'r'
)  
plt.savefig("../seeds/all_seeds/hex_predict_models_merged.png", bbox_inches="tight")
plt.close() 