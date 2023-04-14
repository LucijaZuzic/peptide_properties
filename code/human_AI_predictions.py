import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy import stats
from utils import DATA_PATH
from automate_training import merge_data_TSNE, load_data_SA_TSNE, model_predict_TSNE_seq, model_predict_TSNE_AP_seq, merge_data_AP, merge_data_seq, merge_data, model_predict, model_predict_AP, model_predict_seq, load_data_SA_AP, load_data_SA_seq, load_data_SA
df = pd.read_csv(DATA_PATH + "human_AI.csv", sep = ";")
plt.rcParams.update({'font.size': 22})

def myfunc(x):
  return slope * x + intercept

'''
other_output = open(
    "../final_all/human_AI_predict.txt",
    "r",
    encoding="utf-8",
) 
model_predictions_human_AI = eval(other_output.readlines()[0])
other_output.close()

other_output = open(
    "../final_AP/human_AI_predict.txt",
    "r",
    encoding="utf-8",
)
model_predictions_AP_human_AI = eval(other_output.readlines()[0])
other_output.close()

other_output = open(
    "../final_seq/human_AI_predict.txt",
    "r",
    encoding="utf-8",
)
model_predictions_seq_human_AI = eval(other_output.readlines()[0])
other_output.close()
'''

dict_human_AI = {}
for i in range(len(df['pep'])):
    dict_human_AI[df['pep'][i]] = str(df['agg'][i])

actual_AP = []
for i in df['AP']:
    actual_AP.append(float(i)) 

seq_example = ''
for i in range(24):
    seq_example += 'A'
dict_human_AI[seq_example] = '1' 

best_batch_size = 600
best_model = ''  
NUM_TESTS = 5

offset = 1 
  
properties = np.ones(95)
masking_value = 2

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
    
best_model_file, best_model_image  = "../final_seq/model.h5", "../final_seq/model_picture.png"
 
model_predictions_seq_human_AI = model_predict_seq(best_batch_size, all_data_SEQ, all_labels_SEQ, best_model_file, best_model)[:-1]

other_output = open(
    "../final_seq/human_AI_predict.txt",
    "w",
    encoding="utf-8",
)
other_output.write(str(model_predictions_seq_human_AI))
other_output.write("\n") 
other_output.close() 

names = []
num_props = len(names) * 3
SA, NSA = load_data_SA_TSNE(dict_human_AI, names, offset, masking_value)
all_data_TSNE_seq, all_labels_TSNE_seq = merge_data_TSNE(SA, NSA) 
print('encoded TSNE seq')

best_model_file, best_model_image  = "../final_TSNE_seq/model.h5", "../final_TSNE_seq/model_picture.png"

model_predictions_TSNE_seq_human_AI = model_predict_TSNE_seq(best_batch_size, all_data_TSNE_seq, all_labels_TSNE_seq, best_model_file, best_model)[:-1]

other_output = open(
    "../final_TSNE_seq/human_AI_predict.txt",
    "w",
    encoding="utf-8",
)
other_output.write(str(model_predictions_TSNE_seq_human_AI))
other_output.write("\n") 
other_output.close() 
 
names = ['AP']
num_props = len(names) * 3
SA, NSA = load_data_SA_TSNE(dict_human_AI, names, offset, masking_value)
all_data_TSNE_AP_seq, all_labels_TSNE_AP_seq = merge_data_TSNE(SA, NSA) 
print('encoded TSNE AP seq')

best_model_file, best_model_image  = "../final_TSNE_AP_seq/model.h5", "../final_TSNE_AP_seq/model_picture.png"

model_predictions_TSNE_AP_seq_human_AI = model_predict_TSNE_AP_seq(num_props, best_batch_size, all_data_TSNE_AP_seq, all_labels_TSNE_AP_seq, best_model_file, best_model)[:-1]

other_output = open(
    "../final_TSNE_AP_seq/human_AI_predict.txt",
    "w",
    encoding="utf-8",
)
other_output.write(str(model_predictions_TSNE_AP_seq_human_AI))
other_output.write("\n") 
other_output.close() 

'''
other_output = open(
    "../final_TSNE_seq/human_AI_predict.txt",
    "r",
    encoding="utf-8",
)
model_predictions_TSNE_seq_human_AI = eval(other_output.readlines()[0])
other_output.close()

other_output = open(
    "../final_TSNE_AP_seq/human_AI_predict.txt",
    "r",
    encoding="utf-8",
)
model_predictions_TSNE_AP_seq_human_AI = eval(other_output.readlines()[0])
other_output.close()
'''

plt.figure()
plt.title(
    "SP model"
)
plt.xlabel("Predicted self assembly probability")
plt.ylabel("AP")
plt.scatter(
    model_predictions_seq_human_AI,
    actual_AP,
    color = '#2e85ff',
) 
slope, intercept, r, p, std_err = stats.linregress(model_predictions_seq_human_AI, actual_AP)
print("Seq. model R: " + str(r)) 
print("Seq. model corrcoef: " + str(np.corrcoef(model_predictions_seq_human_AI, actual_AP)[0][1])) 
print("Seq. model spearmanr: " + str(stats.spearmanr(model_predictions_seq_human_AI, actual_AP)[0])) 
print("Seq. model R2: " + str(sklearn.metrics.r2_score(model_predictions_seq_human_AI, actual_AP)))  
mymodel = list(map(myfunc, model_predictions_seq_human_AI)) 
plt.plot(
    model_predictions_seq_human_AI,
    mymodel, color = '#ff120a'
)  
plt.savefig("../final_seq/human_AI_predict.png", bbox_inches="tight")
plt.close() 
 
best_model_file, best_model_image  = "../final_AP/model.h5", "../final_AP/model_picture.png" 

model_predictions_AP_human_AI = model_predict_AP(num_props, best_batch_size, all_data_AP, all_labels_AP, best_model_file, best_model)[:-1]

other_output = open(
    "../final_AP/human_AI_predict.txt",
    "w",
    encoding="utf-8",
)
other_output.write(str(model_predictions_AP_human_AI))
other_output.write("\n") 
other_output.close()  

plt.figure()
plt.title(
    "AP model"
)
plt.xlabel("Predicted self assembly probability")
plt.ylabel("AP")
plt.scatter(
    model_predictions_AP_human_AI,
    actual_AP,
    color = '#2e85ff'
)  
slope, intercept, r, p, std_err = stats.linregress(model_predictions_AP_human_AI, actual_AP)
print("AP model R: " + str(r)) 
print("AP model corrcoef: " + str(np.corrcoef(model_predictions_AP_human_AI, actual_AP)[0][1])) 
print("AP model spearmanr: " + str(stats.spearmanr(model_predictions_AP_human_AI, actual_AP)[0])) 
print("AP model R2: " + str(sklearn.metrics.r2_score(model_predictions_AP_human_AI, actual_AP)))  
mymodel = list(map(myfunc, model_predictions_AP_human_AI)) 
plt.plot(
    model_predictions_AP_human_AI,
    mymodel, color = '#ff120a'
)  
plt.savefig("../final_AP/human_AI_predict.png", bbox_inches="tight")
plt.close() 
 
best_model_file, best_model_image  = "../final_all/model.h5", "../final_all/model_picture.png" 

model_predictions_human_AI = model_predict(num_props, best_batch_size, all_data, all_labels, best_model_file, best_model)[:-1]

other_output = open(
    "../final_all/human_AI_predict.txt",
    "w",
    encoding="utf-8",
)
other_output.write(str(model_predictions_human_AI))
other_output.write("\n") 
other_output.close() 

plt.figure()
plt.title(
    "Hybrid AP-SP model"
)
plt.xlabel("Predicted self assembly probability")
plt.ylabel("AP")
plt.scatter(
    model_predictions_human_AI,
    actual_AP,
    color = '#2e85ff'
)  
slope, intercept, r, p, std_err = stats.linregress(model_predictions_human_AI, actual_AP)
print("Seq. and AP model R: " + str(r)) 
print("Seq. and AP model corrcoef: " + str(np.corrcoef(model_predictions_human_AI, actual_AP)[0][1])) 
print("Seq. and AP model spearmanr: " + str(stats.spearmanr(model_predictions_human_AI, actual_AP)[0])) 
print("Seq. and AP model R2: " + str(sklearn.metrics.r2_score(model_predictions_human_AI, actual_AP)))   
mymodel = list(map(myfunc, model_predictions_human_AI)) 
plt.plot(
    model_predictions_human_AI,
    mymodel, color = '#ff120a'
)  
plt.savefig("../final_all/human_AI_predict.png", bbox_inches="tight")
plt.close() 

plt.figure(figsize=(25, 5))
plt.subplot(1, 3, 1)
plt.title(
    "AP model"
)
plt.xlabel("Predicted self assembly probability")
plt.ylabel("AP")
plt.xticks([0.25, 0.5, 0.75])
plt.scatter(
    model_predictions_AP_human_AI,
    actual_AP,
    color = '#2e85ff'
) 
slope, intercept, r, p, std_err = stats.linregress(model_predictions_AP_human_AI, actual_AP)
mymodel = list(map(myfunc, model_predictions_AP_human_AI)) 
plt.plot(
    model_predictions_AP_human_AI,
    mymodel, color = '#ff120a'
)  
plt.subplot(1, 3, 2)
plt.title(
    "SP model"
)
plt.xlabel("Predicted self assembly probability")
plt.yticks([])
plt.xticks([0.25, 0.5, 0.75])
plt.scatter(
    model_predictions_seq_human_AI,
    actual_AP,
    color = '#2e85ff'
)
slope, intercept, r, p, std_err = stats.linregress(model_predictions_seq_human_AI, actual_AP)
mymodel = list(map(myfunc, model_predictions_seq_human_AI)) 
plt.plot(
    model_predictions_seq_human_AI,
    mymodel, color = '#ff120a'
)  
plt.subplot(1, 3, 3)
plt.title(
    "Hybrid AP-SP model"
)
plt.xlabel("Predicted self assembly probability")
plt.yticks([])
plt.xticks([0.25, 0.5, 0.75])
plt.scatter(
    model_predictions_human_AI,
    actual_AP,
    color = '#2e85ff'
) 
slope, intercept, r, p, std_err = stats.linregress(model_predictions_human_AI, actual_AP)
mymodel = list(map(myfunc, model_predictions_human_AI)) 
plt.plot(
    model_predictions_human_AI,
    mymodel, color = '#ff120a'
)  
plt.savefig("../seeds/all_seeds/human_AI_predict_models_merged.png", bbox_inches="tight")
plt.close() 

plt.figure()
plt.title(
    "t-SNE SP model"
)
plt.xlabel("Predicted self assembly probability")
plt.ylabel("AP")
plt.scatter(
    model_predictions_TSNE_seq_human_AI,
    actual_AP,
    color = '#2e85ff'
)  
slope, intercept, r, p, std_err = stats.linregress(model_predictions_TSNE_seq_human_AI, actual_AP)
print("t-SNE Seq. model R: " + str(r)) 
print("t-SNE Seq. model corrcoef: " + str(np.corrcoef(model_predictions_TSNE_seq_human_AI, actual_AP)[0][1])) 
print("t-SNE Seq. model spearmanr: " + str(stats.spearmanr(model_predictions_TSNE_seq_human_AI, actual_AP)[0])) 
print("t-SNE Seq. model R2: " + str(sklearn.metrics.r2_score(model_predictions_TSNE_seq_human_AI, actual_AP)))   
mymodel = list(map(myfunc, model_predictions_TSNE_seq_human_AI)) 
plt.plot(
    model_predictions_TSNE_seq_human_AI,
    mymodel, color = '#ff120a'
)  
plt.savefig("../final_TSNE_seq/human_AI_predict.png", bbox_inches="tight")
plt.close() 

plt.figure()
plt.title(
    "t-SNE AP-SP model"
)
plt.xlabel("Predicted self assembly probability")
plt.ylabel("AP")
plt.scatter(
    model_predictions_TSNE_AP_seq_human_AI,
    actual_AP,
    color = '#2e85ff'
)  
slope, intercept, r, p, std_err = stats.linregress(model_predictions_TSNE_AP_seq_human_AI, actual_AP)
print("t-SNE Seq. and AP model R: " + str(r)) 
print("t-SNE Seq. and AP model corrcoef: " + str(np.corrcoef(model_predictions_TSNE_AP_seq_human_AI, actual_AP)[0][1])) 
print("t-SNE Seq. and AP model spearmanr: " + str(stats.spearmanr(model_predictions_TSNE_AP_seq_human_AI, actual_AP)[0])) 
print("t-SNE Seq. and AP model R2: " + str(sklearn.metrics.r2_score(model_predictions_TSNE_AP_seq_human_AI, actual_AP)))   
mymodel = list(map(myfunc, model_predictions_TSNE_AP_seq_human_AI)) 
plt.plot(
    model_predictions_TSNE_AP_seq_human_AI,
    mymodel, color = '#ff120a'
)  
plt.savefig("../final_TSNE_AP_seq/human_AI_predict.png", bbox_inches="tight")
plt.close() 

plt.figure(figsize=(25, 5))
plt.subplot(1, 3, 1)
plt.title(
    "AP model"
)
plt.xlabel("Predicted self assembly probability")
plt.ylabel("AP")
plt.xticks([0.25, 0.5, 0.75])
plt.scatter(
    model_predictions_AP_human_AI,
    actual_AP,
    color = '#2e85ff'
) 
slope, intercept, r, p, std_err = stats.linregress(model_predictions_AP_human_AI, actual_AP)
mymodel = list(map(myfunc, model_predictions_AP_human_AI)) 
plt.plot(
    model_predictions_AP_human_AI,
    mymodel, color = '#ff120a'
)  
plt.subplot(1, 3, 2)
plt.title(
    "t-SNE SP model"
)
plt.xlabel("Predicted self assembly probability")
plt.yticks([])
plt.xticks([0.25, 0.5, 0.75])
plt.scatter(
    model_predictions_TSNE_seq_human_AI,
    actual_AP,
    color = '#2e85ff'
)
slope, intercept, r, p, std_err = stats.linregress(model_predictions_TSNE_seq_human_AI, actual_AP)
mymodel = list(map(myfunc, model_predictions_TSNE_seq_human_AI)) 
plt.plot(
    model_predictions_TSNE_seq_human_AI,
    mymodel, color = '#ff120a'
)  
plt.subplot(1, 3, 3)
plt.title(
    "t-SNE AP-SP model"
)
plt.xlabel("Predicted self assembly probability")
plt.yticks([])
plt.xticks([0.25, 0.5, 0.75])
plt.scatter(
    model_predictions_TSNE_AP_seq_human_AI,
    actual_AP,
    color = '#2e85ff'
) 
slope, intercept, r, p, std_err = stats.linregress(model_predictions_TSNE_AP_seq_human_AI, actual_AP)
mymodel = list(map(myfunc, model_predictions_TSNE_AP_seq_human_AI)) 
plt.plot(
    model_predictions_TSNE_AP_seq_human_AI,
    mymodel, color = '#ff120a'
)  
plt.savefig("../seeds/all_seeds/TSNE_human_AI_predict_models_merged.png", bbox_inches="tight")
plt.close() 

plt.figure(figsize=(25, 5))
plt.subplot(1, 3, 1)
plt.title(
    "AP model"
)
plt.xlabel("Predicted self assembly probability")
plt.ylabel("AP")
plt.xticks([0.25, 0.5, 0.75])
plt.scatter(
    model_predictions_AP_human_AI,
    actual_AP,
    color = '#2e85ff'
) 
slope, intercept, r, p, std_err = stats.linregress(model_predictions_AP_human_AI, actual_AP)
mymodel = list(map(myfunc, model_predictions_AP_human_AI)) 
plt.plot(
    model_predictions_AP_human_AI,
    mymodel, color = '#ff120a'
)  
plt.subplot(1, 3, 2)
plt.title(
    "SP model"
)
plt.xlabel("Predicted self assembly probability")
plt.yticks([])
plt.xticks([0.25, 0.5, 0.75])
plt.scatter(
    model_predictions_seq_human_AI,
    actual_AP,
    color = '#2e85ff'
)
slope, intercept, r, p, std_err = stats.linregress(model_predictions_seq_human_AI, actual_AP)
mymodel = list(map(myfunc, model_predictions_seq_human_AI)) 
plt.plot(
    model_predictions_seq_human_AI,
    mymodel, color = '#ff120a'
)  
plt.subplot(1, 3, 3)
plt.title(
    "t-SNE AP-SP model"
)
plt.xlabel("Predicted self assembly probability")
plt.yticks([])
plt.xticks([0.25, 0.5, 0.75])
plt.scatter(
    model_predictions_TSNE_AP_seq_human_AI,
    actual_AP,
    color = '#2e85ff'
) 
slope, intercept, r, p, std_err = stats.linregress(model_predictions_TSNE_AP_seq_human_AI, actual_AP)
mymodel = list(map(myfunc, model_predictions_TSNE_AP_seq_human_AI)) 
plt.plot(
    model_predictions_TSNE_AP_seq_human_AI,
    mymodel, color = '#ff120a'
)  
plt.savefig("../seeds/all_seeds/TSNE_YES_NO_human_AI_predict_models_merged.png", bbox_inches="tight")
plt.close() 