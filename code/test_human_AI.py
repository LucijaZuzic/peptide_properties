import pandas as pd
from utils import DATA_PATH, SEQ_MODEL_DATA_PATH, MODEL_DATA_PATH, MY_MODEL_DATA_PATH, TSNE_AP_SEQ_DATA_PATH, TSNE_SEQ_DATA_PATH, predictions_name, setSeed

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
dict_AI = {}
dict_human = {}
for i in range(len(df['pep'])):
    dict_human_AI[df['pep'][i]] = str(df['agg'][i])
    if df['expert'][i] == 'Human':
        dict_human[df['pep'][i]] = str(df['agg'][i])
    else:
        dict_AI[df['pep'][i]] = str(df['agg'][i])

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
 
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276] 
NUM_TESTS = 5

dict_human_AI_for_seeds = {seed: {} for seed in seed_list}
dict_human_for_seeds = {seed: {} for seed in seed_list}
dict_AI_for_seeds = {seed: {} for seed in seed_list}
 
alias_for_name = [MY_MODEL_DATA_PATH, SEQ_MODEL_DATA_PATH, MODEL_DATA_PATH, TSNE_SEQ_DATA_PATH, TSNE_AP_SEQ_DATA_PATH]  

for seed in seed_list: 
    print(seed)
    setSeed(seed)
    for number in range(1, NUM_TESTS + 1): 
        print(number)
        
        test_name = "../seeds/seed_" + str(seed) + "/similarity/" + 'test_fold_' + str(number) + ".csv"
        test_name_file = open(test_name, "r")
        test_name_lines = test_name_file.readlines()
        test_name_file.close()

        test_name_lines = test_name_lines[1:]
        for i in range(len(test_name_lines)):
            test_name_lines[i] = test_name_lines[i].split(",")[0]

        some_model_output_line = []
        for i in range(len(alias_for_name)):
            some_model_output = open(
                predictions_name(alias_for_name[i], number, 'weak', 1),
                "r",
                encoding="utf-8",
            )
            some_model_output_line.append(eval(some_model_output.readlines()[0]))
            some_model_output.close()
  
        print(test_name_lines[1])
        for pep in dict_human_AI:
            if pep in test_name_lines:
                ind = test_name_lines.index(pep)
                retval = ""
                for i in range(len(some_model_output_line)):
                    retval += ";" + str(some_model_output_line[i][ind]) 
                dict_human_AI_for_seeds[seed][pep] = retval
                if pep in dict_human:
                    dict_human_for_seeds[seed][pep] = retval
                else:
                    dict_AI_for_seeds[seed][pep] = retval
    print(dict_human_AI_for_seeds[seed])
    print(dict_human_for_seeds[seed])
    print(dict_AI_for_seeds[seed])

    all_lines_new_table = lines_new_table[0]  
    for pep in dict_human_AI: 
        all_lines_new_table += pep + dict_human_AI_for_seeds[seed][pep] + "\n"
    print(all_lines_new_table)

    file_new_table = open(DATA_PATH + "human_AI_results_folds_" + str(seed) + ".csv", "w")
    file_new_table.write(all_lines_new_table)
    file_new_table.close()

    all_lines_new_table = lines_new_table[0]  
    for pep in dict_human: 
        all_lines_new_table += pep + dict_human_AI_for_seeds[seed][pep] + "\n"
    print(all_lines_new_table)

    file_new_table = open(DATA_PATH + "human_results_folds_" + str(seed) + ".csv", "w")
    file_new_table.write(all_lines_new_table)
    file_new_table.close()

    all_lines_new_table = lines_new_table[0]  
    for pep in dict_AI: 
        all_lines_new_table += pep + dict_human_AI_for_seeds[seed][pep] + "\n"
    print(all_lines_new_table)

    file_new_table = open(DATA_PATH + "AI_results_folds_" + str(seed) + ".csv", "w")
    file_new_table.write(all_lines_new_table)
    file_new_table.close()

for seed in seed_list: 
    df = pd.read_csv(DATA_PATH + "AI_results_folds_" + str(seed) + ".csv", sep = ";")
    res = df['AP']
    print(sum(res))