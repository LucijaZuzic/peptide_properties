import pandas as pd

from utils import DATA_PATH

df = pd.read_csv(DATA_PATH + "human_AI.csv", sep = ";") 

new = ['VVVVV', 
'FKFEF',
'VKVEV',
'VKVFF',
'KFFFE',
'KFAFD',
'RVSVD',
'KKFDD',
'VKVKV',
'KVKVK',
'DPDPD',
'SYCGY',
'FFEKF',
'RWLDY',
'KWEFY',
'FKIDF',
'KWMDF',
'WKPYY',
'PPPHY',
'PTPCY',
'FFEKF',
'KFFDY',
'KDHFY',
'YTEYK',
'WKPYY',
'EPYYK',
'YDPKY',
'KDPYY',
'YEPYK']

grades = [1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0]

dict_human_AI = {}
for i in range(len(df['pep'])):
    dict_human_AI[df['pep'][i]] = str(df['agg'][i])

actual_AP = []
for i in df['AP']:
    actual_AP.append(i) 

for i in range(len(new)):
    if new[i] in dict_human_AI:
        print("YES", new[i], grades[i], dict_human_AI[new[i]])
    else:
        print("NO", new[i], grades[i])

for x in dict_human_AI:
    if x not in new:
        print("ERROR", x)


paths = ["../final_AP/human_AI_predict.txt", "../final_seq/human_AI_predict.txt", "../final_all/human_AI_predict.txt", 
        "../final_TSNE_seq/human_AI_predict.txt", "../final_TSNE_AP_seq/human_AI_predict.txt"] 
names = ["AP", "SP", "Hybrid AP-SP",  "t-SNE SP", "t-SNE AP-SP"]  
 
lines_new_table = ["pep"]

for name in names:
    lines_new_table[0] += ";" + name

for pep in dict_human_AI:
    lines_new_table.append(pep) 

ind = -1
for some_path in paths: 
    ind += 1

    model_predictions_human_AI_file = open(some_path, 'r')

    model_predictions_human_AI_lines = model_predictions_human_AI_file.readlines()

    model_predictions_human_AI_one = eval(model_predictions_human_AI_lines[0])

    print(model_predictions_human_AI_one)

    for i in range(len(model_predictions_human_AI_one)):
        lines_new_table[i + 1] += ";" + str(model_predictions_human_AI_one[i])

all_lines_new_table = ""
for i in range(len(lines_new_table)):
    lines_new_table[i] += "\n"
    all_lines_new_table += lines_new_table[i]
print(all_lines_new_table)

file_new_table = open(DATA_PATH + "human_AI_results.csv", "w")
file_new_table.write(all_lines_new_table)
file_new_table.close()