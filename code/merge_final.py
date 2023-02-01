from utils import PATH_TO_NAME, predictions_name, final_history_name, DATA_PATH, SEQ_MODEL_DATA_PATH, MODEL_DATA_PATH, MY_MODEL_DATA_PATH, setSeed, getSeed, PATH_TO_EXTENSION
from custom_plots import merge_type_iteration, results_name,my_accuracy_calculate, weird_division, convert_to_binary
import matplotlib.pyplot as plt
import os
import numpy as np 
import seaborn as sns
import pandas as pd

def read_one_final_history(some_path, test_number, iteration):
    acc_path, loss_path = final_history_name(some_path, test_number, iteration)
    acc_file = open(acc_path, "r")
    acc_lines = acc_file.readlines()
    acc = eval(acc_lines[0]) 
    acc_file.close() 
    loss_file = open(loss_path, "r")
    loss_lines = loss_file.readlines()
    loss = eval(loss_lines[0]) 
    loss_file.close() 
    return acc, loss  

def read_all_final_history(some_path, iteration, lines_dict, sd_dict): 
    all_acc = []
    all_loss = []
    for test_number in range(1, NUM_TESTS + 1):
        acc, loss = read_one_final_history(some_path, test_number, iteration)
        for a in acc:
            all_acc.append(float(a))
        for l in loss:
            all_loss.append(float(l))

    lines_dict['Maximum accuracy = '].append(np.max(all_acc) * 100)
    lines_dict['Minimal loss = '].append(np.min(all_loss) * 100)
    lines_dict['Accuracy = '].append(np.mean(all_acc) * 100)
    lines_dict['Loss = '].append(np.mean(all_loss))
    sd_dict['Accuracy = '].append(np.std(all_acc) * 100)
    sd_dict['Loss = '].append(np.std(all_loss)) 

def read_one_prediction(some_path, test_number, final_model_type, iteration):
    file = open(predictions_name(some_path, test_number, final_model_type, iteration), "r")
    lines = file.readlines()
    predictions = eval(lines[0])
    labels = eval(lines[1])
    file.close()
    return predictions, labels 

def read_all_model_predictions(some_path, min_test_number, max_test_number, final_model_type, iteration):
    all_predictions = []
    all_labels = []
    for test_number in range(min_test_number, max_test_number + 1): 
        predictions, labels = read_one_prediction(some_path, test_number, final_model_type, iteration)
        for prediction in predictions:
            all_predictions.append(prediction)
        for label in labels:
            all_labels.append(label) 
    return all_predictions, all_labels

def hist_predicted_merged(
    model_type,
    test_number,
    final_model_type,
    iteration,
    test_labels,
    model_predictions,
    save
):
    # Create a histogram of the predicted probabilities only for the peptides that show self-assembly

    model_predictions_true = []
    model_predictions_false = []
    for x in range(len(test_labels)):
        if test_labels[x] == 1.0:
            model_predictions_true.append(float(model_predictions[x]))
        else:
            model_predictions_false.append(float(model_predictions[x]))

    plt.figure()
    # Draw the density plot
    sns.displot({"SA": model_predictions_true}, kde=True, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], palette = {"SA": '#0485d1'}, legend = False)
    plt.ylim(0, 750)
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides")
    plt.title(
        merge_type_iteration(model_type, final_model_type, iteration, test_number).replace("Test 0 Weak 1", "")
    )
    plt.savefig(save + "_SA.png", bbox_inches="tight")
    plt.close()

    # Create a histogram of the predicted probabilities only for the peptides that don't show self-assembly

    plt.figure()
    sns.displot({"NSA": model_predictions_false}, kde=True, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], palette = {"NSA": 'red'}, legend = False)
    plt.ylim(0, 750)
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides") 
    plt.title(
        merge_type_iteration(model_type, final_model_type, iteration, test_number).replace("Test 0 Weak 1", "")
    )
    plt.savefig(save + "_NSA.png", bbox_inches="tight")
    plt.close() 

    plt.figure()
    sns.displot({"SA": model_predictions_true, "NSA": model_predictions_false}, kde=True, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], palette = {"SA": '#0485d1', "NSA": 'red'})
    plt.ylim(0, 750)
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides") 
    plt.title(
        merge_type_iteration(model_type, final_model_type, iteration, test_number).replace("Test 0 Weak 1", "")
    )
    plt.savefig(save + "_all.png", bbox_inches="tight")
    plt.close() 

def read_one_result(some_path, test_number):       
    file = open(results_name(some_path, test_number), "r") 
    lines = file.readlines() 
    file.close()
    return lines

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
paths = [SEQ_MODEL_DATA_PATH, MODEL_DATA_PATH, MY_MODEL_DATA_PATH]
NUM_TESTS = 5
all_preds = []
all_labels_new= []
all_model_types = []
for some_path in paths:
    seed_predictions = []
    seed_labels = []
    for seed in seed_list:
        setSeed(seed)
        all_predictions, all_labels = read_all_model_predictions(some_path, 1, 5, "weak", 1)
        for pred in all_predictions:
            seed_predictions.append(pred)
            all_preds.append(pred)
        for label in all_labels:
            seed_labels.append(label)
            if label == 0.0:
                all_labels_new.append('NSA') 
            else:
                all_labels_new.append('SA') 
            all_model_types.append(PATH_TO_NAME[some_path])

    hist_predicted_merged(
        some_path,
        0,
        "weak",
        1,
        seed_labels,
        seed_predictions,
        "../seeds/all_seeds/" + PATH_TO_EXTENSION[some_path] + "_hist_merged_seeds"
    ) 

#increase font size of all elements
sns.set(font_scale=1.5)
d = {'Predicted self assembly probability': all_preds, 'Self assembly status': all_labels_new, 'Model': all_model_types}
df = pd.DataFrame(data=d)
plt.figure()
g = sns.displot(data=df, x = 'Predicted self assembly probability', kde=True, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], hue = 'Self assembly status', col = 'Model', palette = {'NSA': '#0485d1', 'SA': 'red'})
g.set_axis_labels("Predicted self assembly probability", "Number of peptides")
g.set_titles("Model {col_name}")  
plt.savefig("../seeds/all_seeds/hist_merged_models.png", bbox_inches="tight")
plt.close() 