from utils import TSNE_AP_SEQ_DATA_PATH, TSNE_SEQ_DATA_PATH, PATH_TO_NAME, predictions_name, final_history_name, SEQ_MODEL_DATA_PATH, MODEL_DATA_PATH, MY_MODEL_DATA_PATH, setSeed, PATH_TO_EXTENSION
from custom_plots import merge_type_iteration, results_name, weird_division
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve
)

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
    sns.displot({"SA": model_predictions_true}, kde=True, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], palette = {"SA": '#2e85ff'}, legend = False)
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
    sns.displot({"NSA": model_predictions_false}, kde=True, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], palette = {"NSA": '#ff120a'}, legend = False)
    plt.ylim(0, 750)
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides") 
    plt.title(
        merge_type_iteration(model_type, final_model_type, iteration, test_number).replace("Test 0 Weak 1", "")
    )
    plt.savefig(save + "_NSA.png", bbox_inches="tight")
    plt.close() 

    plt.figure()
    sns.displot({"SA": model_predictions_true, "NSA": model_predictions_false}, kde=True, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], palette = {"SA": '#2e85ff', "NSA": '#ff120a'})
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

#increase font size of all elements
#sns.set(font_scale=1.5)
plt.rcParams.update({'font.size': 22})

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
paths = [SEQ_MODEL_DATA_PATH, MODEL_DATA_PATH, MY_MODEL_DATA_PATH, TSNE_SEQ_DATA_PATH, TSNE_AP_SEQ_DATA_PATH]
NUM_TESTS = 5
all_preds = []
all_labels_new = []
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
            all_model_types.append(PATH_TO_NAME[some_path].replace("SP and AP", "Hybrid AP-SP").replace("TSNE", "t-SNE"))

    hist_predicted_merged(
        some_path,
        0,
        "weak",
        1,
        seed_labels,
        seed_predictions,
        "../seeds/all_seeds/" + PATH_TO_EXTENSION[some_path] + "_hist_merged_seeds"
    ) 

APpreds = []
SPpreds =  []
SPAPpreds = []
TSNESPpreds =  []
TSNESPAPpreds = []
APN = []
SPN = []
SPAPN = []
TSNESPN = []
TSNESPAPN = []
APY = []
SPY= []
SPAPY = []
TSNESPY = []
TSNESPAPY = []

for i in range(len(all_preds)):
    if all_model_types[i] == 'AP':
        APpreds.append(all_preds[i])
        if all_labels_new[i] == 'SA':
            APY.append(all_preds[i])
        else:
            APN.append(all_preds[i])
        continue
    if all_model_types[i] == 'SP':
        SPpreds.append(all_preds[i])
        if all_labels_new[i] == 'SA':
            SPY.append(all_preds[i])
        else:
            SPN.append(all_preds[i])
        continue
    if all_model_types[i] == 'TSNE SP':
        TSNESPpreds.append(all_preds[i])
        if all_labels_new[i] == 'SA':
            TSNESPY.append(all_preds[i])
        else:
            TSNESPN.append(all_preds[i])
        continue
    if all_model_types[i] == 'TSNE AP-SP':
        TSNESPAPpreds.append(all_preds[i])
        if all_labels_new[i] == 'SA':
            TSNESPAPY.append(all_preds[i])
        else:
            TSNESPAPN.append(all_preds[i])
        continue 
    SPAPpreds.append(all_preds[i])
    if all_labels_new[i] == 'SA':
        SPAPY.append(all_preds[i])
    else:
        SPAPN.append(all_preds[i]) 

d = {'Predicted self assembly probability': all_preds, 'Self assembly status': all_labels_new, 'Model': all_model_types}
df = pd.DataFrame(data=d)
plt.figure()
g = sns.displot(data=df, x = 'Predicted self assembly probability', kde=True, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], hue = 'Self assembly status', col = 'Model', palette = {'NSA': '#ff120a', 'SA': '#2e85ff'})
g.set_axis_labels("Self assembly probability", "Number of peptides")
g.set_titles("{col_name} model")  
plt.savefig("../seeds/all_seeds/hist_merged_models.png", bbox_inches="tight")
plt.close() 

'''
plt.figure(figsize=(25, 5))

plt.subplot(1, 3, 1)
sns.displot({"SA": APY, "NSA": APN}, kde=True, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], palette = {"SA": '#2e85ff', "NSA": '#ff120a'})
plt.xlabel("Predicted self assembly probability")
plt.ylabel("Number of peptides") 
plt.title("AP model")   

plt.subplot(1, 3, 2)
sns.displot({"SA": SPY, "NSA": SPN}, kde=True, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], palette = {"SA": '#2e85ff', "NSA": '#ff120a'})
plt.xlabel("Predicted self assembly probability")
plt.ylabel("Number of peptides")
plt.title("SP model")  

plt.subplot(1, 3, 3)
sns.displot({"SA": SPAPY, "NSA": SPAPN}, kde=True, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], palette = {"SA": '#2e85ff', "NSA": '#ff120a'})
plt.xlabel("Predicted self assembly probability")
plt.ylabel("Number of peptides")
plt.title("Hybrid AP-SP model")  

plt.savefig("../seeds/all_seeds/hist_merged_models_new.png", bbox_inches="tight")
plt.close()  
'''

def read_ROC(test_labels, model_predictions, name): 
    # Get false positive rate and true positive rate.
    fpr, tpr, thresholds = roc_curve(test_labels, model_predictions) 

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ix = np.argmax(gmeans) 
     
    plt.figure()
    #plt.title(
    #    name + " model"
    #    + "\nReceiver operating characteristic (ROC) curve"
    #)

    plt.axvline(fpr[ix], linestyle = '--', color = 'y')
    plt.axhline(tpr[ix],  linestyle = '--', color = 'y')

    radius = np.sqrt(np.power(fpr[ix], 2) + np.power(1 - tpr[ix], 2))
    circle1 = plt.Circle((0, 1), radius, color = '#2e85ff')
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_patch(circle1)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    plt.arrow(fpr[ix], tpr[ix], - fpr[ix], 1 - tpr[ix], length_includes_head = True, head_width = 0.02)
 
    # Plot ROC curve.
    plt.plot(fpr, tpr, "r", label="ROC curve")
    plt.plot(fpr[ix], tpr[ix], "o", markerfacecolor="r", markeredgecolor="k")

    # Plot random guessing ROC curve.
    plt.plot([0, 1], [0, 1], "c", label="ROC curve for random guessing")

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.savefig(
        '../seeds/all_seeds/' + name + '_ROC.png',
        bbox_inches="tight",
    )

    plt.close()
   
def read_PR(test_labels, model_predictions, name):  
    # Get recall and precision.
    precision, recall, thresholds = precision_recall_curve(
        test_labels, model_predictions
    ) 

    # Calculate the F1 score for each threshold
    fscore = []
    for i in range(len(precision)):
        fscore.append(
            weird_division(2 * precision[i] * recall[i], precision[i] + recall[i])
        )

    # Locate the index of the largest F1 score
    ix = np.argmax(fscore)
    
    plt.figure()
    #plt.title(
    #    name + " model"
    #    + "\nPrecision - Recall (PR) curve"
    #)  

    plt.axvline(recall[ix], linestyle = '--', color = 'y')
    plt.axhline(precision[ix],  linestyle = '--', color = 'y')

    radius = np.sqrt(np.power(1 - recall[ix], 2) + np.power(1 - precision[ix], 2))
    circle1 = plt.Circle((1, 1), radius, color = '#2e85ff')
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_patch(circle1)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    plt.arrow(recall[ix], precision[ix], 1 - recall[ix], 1 - precision[ix], length_includes_head = True, head_width = 0.02)
 
    # Plot PR curve.
    plt.plot(recall, precision, "r", label="PR curve")
    plt.plot(
        recall[ix], precision[ix], "o", markerfacecolor="r", markeredgecolor="k"
    )
 
    # Calculate the no skill line as the proportion of the positive class
    num_positive = 0
    for value in test_labels:
        if value == 1:
            num_positive += 1
    no_skill = num_positive / len(test_labels)

    # Plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], "c", label="PR curve for random guessing")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.savefig(
        '../seeds/all_seeds/' + name + '_PR.png',
        bbox_inches="tight",
    )

    plt.close()

names = ["SP", "Hybrid AP-SP", "AP", "t-SNE SP", "t-SNE AP-SP"]  
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
paths = [SEQ_MODEL_DATA_PATH, MODEL_DATA_PATH, MY_MODEL_DATA_PATH, TSNE_SEQ_DATA_PATH, TSNE_AP_SEQ_DATA_PATH]
NUM_TESTS = 5
all_preds = []
all_labels_new= []
all_model_types = []
ind = 0
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
            all_model_types.append(PATH_TO_NAME[some_path].replace("SP and AP", "Hybrid AP-SP").replace("TSNE", "t-SNE"))
    
    read_PR(seed_labels, seed_predictions, names[ind])
    read_ROC(seed_labels, seed_predictions, names[ind])

    ind += 1