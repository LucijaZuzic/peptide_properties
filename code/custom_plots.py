from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, auc, f1_score
import matplotlib.pyplot as plt
import numpy as np
from utils import MODEL_DATA_PATH, MERGE_MODEL_DATA_PATH

# Plot the history for training a model
def plt_model(test_number, history, model_file_name, data_to_load="AP", merge = False):
    
    # Summarize history for accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='točnost')
    plt.plot(history.history['val_accuracy'], label='točnost prilikom validacije')
    plt.title('Postotak točnih predviđanja')
    plt.ylabel('točnost')
    plt.xlabel('epoha')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2)
    if not merge:
       plt.savefig(MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+model_file_name+"_acc.png", bbox_inches='tight')
    else: 
       plt.savefig(MERGE_MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+model_file_name+"_acc.png", bbox_inches='tight')
    plt.close()
    # Summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'], label='gubitak')
    plt.plot(history.history['val_loss'], label='gubitak prilikom validacije')
    plt.title('Funkcija gubitka\n(Entropija razlike između točne i predviđene klase)')
    plt.ylabel('gubitak')
    plt.xlabel('epoha')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2)
    if not merge:
       plt.savefig(MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+model_file_name+"_loss.png", bbox_inches='tight')
    else: 
       plt.savefig(MERGE_MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+model_file_name+"_loss.png", bbox_inches='tight')
    plt.close()

# Convert probability to class based on the threshold of probability
def convert_to_binary(model_predictions, threshold=0.5):
    
    model_predictions_binary = []

    for x in model_predictions:
        if x >= threshold:
            model_predictions_binary.append(1.0)
        else:
            model_predictions_binary.append(0.0)

    return model_predictions_binary

def weird_division(n, d):
    return n / d if d else 0

def make_a_PR_plot(test_labels, model_predictions, pattern, label):

   # Get recall and precision.
    precision, recall, thresholds = precision_recall_curve(test_labels, model_predictions)

    # Plot PR curve.
    plt.plot(recall, precision, pattern, label=label)

    # Calculate the F1 score for each threshold
    fscore = []
    for i in range(len(precision)):
        fscore.append(weird_division(2 * precision[i] * recall[i], precision[i] + recall[i]))

    # Locate the index of the largest F1 score
    ix = np.argmax(fscore)
    print(label+': Najbolji prag=%f, F1=%.3f' % (thresholds[ix], fscore[ix]))
    plt.plot(recall[ix], precision[ix], 'o', markerfacecolor=pattern, markeredgecolor='k')

def make_PR_plots(test_number, test_labels, PR_file_name, data_to_load="AP", merge=False, model_predictions_amino=[], model_predictions_di=[], model_predictions_tri=[], model_predictions_ansamble=[], model_predictions_voting=[], model_predictions_avg=[]):
    
    plt.figure()
    plt.title("Krivulja preciznosti i osjetljivosti (PR)")

    if len(model_predictions_amino) != 0:
        make_a_PR_plot(test_labels, model_predictions_amino, 'r', 'PR krivulja za model aminokiselina ' + data_to_load)
        
    if len(model_predictions_di) != 0:
        make_a_PR_plot(test_labels, model_predictions_di, 'g', 'PR krivulja za model dipeptida ' + data_to_load)

    if len(model_predictions_tri) != 0:
        make_a_PR_plot(test_labels, model_predictions_tri, 'b', 'PR krivulja za mode tripeptida ' + data_to_load)

    if len(model_predictions_ansamble) != 0:
        make_a_PR_plot(test_labels, model_predictions_ansamble, 'y', 'PR krivulja za model ansambla ' + data_to_load)

    if len(model_predictions_voting) != 0:
        make_a_PR_plot(test_labels, model_predictions_voting, 'm', 'PR krivulja za model glasovanja ' + data_to_load)

    if len(model_predictions_avg) != 0:
        make_a_PR_plot(test_labels, model_predictions_avg, 'orange', 'PR krivulja za model prosjeka ' + data_to_load)

	# Calculate the no skill line as the proportion of the positive class
    num_positive = 0
    for value in test_labels:
        if value == 1:
            num_positive += 1
    no_skill = num_positive / len(test_labels)

	# Plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], 'c', label='PR krivulja nasumičnog pogađanja')
    plt.xlabel('Osjetljivost')
    plt.ylabel('Preciznost')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2)
    if merge:
        plt.savefig(MERGE_MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+PR_file_name+"_merged_"+data_to_load+".png", bbox_inches = 'tight')
    else:
        plt.savefig(MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+PR_file_name+"_"+data_to_load+".png", bbox_inches = 'tight')
    plt.close()

def make_a_ROC_plot(test_labels, model_predictions, pattern, label):

    plt.title("Krivulja radnih karakteristika prijemnika (ROC)")
    
    # Get false positive rate and true positive rate.
    fpr, tpr, thresholds = roc_curve(test_labels, model_predictions)

    # Plot ROC curve.
    plt.plot(fpr, tpr, pattern, label=label)

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))

    # Locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print(label+': Najbolji prag=%f, Geometrijska sredina=%.3f' % (thresholds[ix], gmeans[ix]))
    plt.plot(fpr[ix], tpr[ix], 'o', markerfacecolor=pattern, markeredgecolor='k')

def make_ROC_plots(test_number, test_labels, ROC_file_name, data_to_load="AP", merge=False, model_predictions_amino=[], model_predictions_di=[], model_predictions_tri=[], model_predictions_ansamble=[], model_predictions_voting=[], model_predictions_avg=[]):
    
    plt.figure()

    if len(model_predictions_amino) != 0:
        make_a_ROC_plot(test_labels, model_predictions_amino, 'r', 'ROC krivulja za model aminokiselina ' + data_to_load)
        
    if len(model_predictions_di) != 0:
        make_a_ROC_plot(test_labels, model_predictions_di, 'g', 'ROC krivulja za model dipeptida ' + data_to_load)

    if len(model_predictions_tri) != 0:
        make_a_ROC_plot(test_labels, model_predictions_tri, 'b', 'ROC krivulja za model tripeptida ' + data_to_load)

    if len(model_predictions_ansamble) != 0:
        make_a_ROC_plot(test_labels, model_predictions_ansamble, 'y', 'ROC krivulja za model ansambla ' + data_to_load)

    if len(model_predictions_voting) != 0:
        make_a_ROC_plot(test_labels, model_predictions_voting, 'm', 'ROC krivulja za model glasovanja ' + data_to_load)
    
    if len(model_predictions_avg) != 0:
        make_a_ROC_plot(test_labels, model_predictions_avg, 'orange', 'ROC krivulja za model prosjeka ' + data_to_load)

    # Plot random guessing ROC curve.
    plt.plot([0, 1], [0, 1], 'c', label='ROC krivulja nasumičnog pogađanja')

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2)
    if merge:
        plt.savefig(MERGE_MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+ROC_file_name+"_merged_"+data_to_load+".png", bbox_inches = 'tight')
    else:
        plt.savefig(MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+ROC_file_name+"_"+data_to_load+".png", bbox_inches = 'tight')
    plt.close()

# Count correct predictions based on a custom threshold of probability
def my_accuracy_calculate(test_labels, model_predictions, threshold=0.5):

    score = 0

    model_predictions = convert_to_binary(model_predictions, threshold)

    for i in range(len(test_labels)):

        if (model_predictions[i] == test_labels[i]):
            score += 1
        
    return score / len(test_labels) * 100

def output_metrics(test_labels, model_predictions, name_of_model, data_to_load="AP", threshold=0.5):
    # Get recall and precision.
    precision, recall, _ = precision_recall_curve(test_labels, model_predictions)

    # Convert probabilities to predictions
    model_predictions_binary = convert_to_binary(model_predictions, threshold)

    print("Model %s %s: Točnost: %.2f%% Površina ispod ROC krivulje: %.4f Površina ispod PR krivulje: %.4f F1: %.4f" 
                        % (name_of_model, data_to_load,
                        my_accuracy_calculate(test_labels, model_predictions, threshold), 
                        roc_auc_score(test_labels, model_predictions), 
                        auc(recall, precision), 
                        f1_score(test_labels, model_predictions_binary)))

def hist_predicted(test_number, test_labels, model_predictions, hist_file_name, data_to_load="AP", merge=False):

    # Create a histogram of the predicted probabilities only for the peptides that show self-assembly

    model_predictions_true = []
    for x in range(len(test_labels)):
        if test_labels[x] == 1.0:
            model_predictions_true.append(model_predictions[x])

    plt.figure()
    plt.title("Model "+hist_file_name+"\nHistogram predviđenih vjerojatnosti samosastavljanja\nza peptide koji imaju samosastavljanje")
    plt.xlabel("Predviđene vjerojatnosti samosastavljanja")
    plt.ylabel("Broj peptida")
    plt.hist(model_predictions_true, bins=100)
    if merge:
        plt.savefig(MERGE_MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+"model_"+hist_file_name+"_merged_"+data_to_load+"_histogram_SA.png", bbox_inches = 'tight')
    else:
        plt.savefig(MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+"model_"+hist_file_name+"_"+data_to_load+"_histogram_SA.png", bbox_inches = 'tight')
    plt.close()

    # Create a histogram of the predicted probabilities only for the peptides that don't show self-assembly

    model_predictions_false = []
    for x in range(len(test_labels)):
        if test_labels[x] == 0.0:
            model_predictions_false.append(model_predictions[x])

    plt.figure()
    plt.title("Model "+hist_file_name+"\nHistogram predviđenih vjerojatnosti samosastavljanja\nza peptide koji nemaju samosastavljanje")
    plt.xlabel("Predviđene vjerojatnosti samosastavljanja")
    plt.ylabel("Broj peptida")
    plt.hist(model_predictions_false, bins=100)
    if merge:
        plt.savefig(MERGE_MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+"model_"+hist_file_name+"_merged_"+data_to_load+"_histogram_NSA.png", bbox_inches = 'tight')
    else:
        plt.savefig(MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+"model_"+hist_file_name+"_"+data_to_load+"_histogram_NSA.png", bbox_inches = 'tight')
    plt.close()

def decorate_stats(model_type, index, history, data_to_load="AP"):
    accuracy = [history[i].history['accuracy'] for i in range(len(history))]
    val_acc = [history[i].history['val_accuracy'] for i in range(len(history))]
    loss = [history[i].history['loss'] for i in range(len(history))]
    val_loss = [history[i].history['val_loss'] for i in range(len(history))]

    accuracy_max = [np.max(accuracy[i]) for i in range(len(history))]
    val_acc_max = [np.max(val_acc[i]) for i in range(len(history))]
    loss_min = [np.min(loss[i]) for i in range(len(history))]
    val_loss_min = [np.min(val_loss[i]) for i in range(len(history))]

    print('Prosjek modela %s %s: Maksimalna točnost=%.2f%% (%.2f%%) Maksimalna točnost pri validaciji=%.2f%% (%.2f%%) Minimalni gubitak=%.2f (%.2f%%) Minimalni gubitak pri validaciji=%.2f (%.2f%%)' 
    % (model_type, data_to_load, np.mean(accuracy_max)*100, np.std(accuracy_max)*100, np.mean(val_acc_max)*100, np.std(val_acc_max)*100, np.mean(loss_min), np.std(loss_min)*100, np.mean(val_loss_min), np.std(val_loss_min)*100))
    print('Prosjek modela %s %s: Točnost=%.2f%% (%.2f%%) Točnost pri validaciji=%.2f%% (%.2f%%) Gubitak=%.2f (%.2f%%) Gubitak pri validaciji=%.2f (%.2f%%)' 
    % (model_type, data_to_load, np.mean(accuracy)*100, np.std(accuracy)*100, np.mean(val_acc)*100, np.std(val_acc)*100, np.mean(loss), np.std(loss)*100, np.mean(val_loss), np.std(val_loss)*100))

    for i in range(len(history)):

        begin_string = "Model"
        if i == index:
            begin_string = "Najbolji model"

        print('%s %s %s (indeks %d): Maksimalna točnost=%.2f%% Maksimalna točnost pri validaciji=%.2f%% Minimalni gubitak=%.2f Minimalni gubitak pri validaciji=%.2f' 
        % (begin_string, model_type, data_to_load, i + 1, accuracy_max[i]*100, val_acc_max[i]*100, loss_min[i], val_loss_min[i]))
        print('%s %s %s (indeks %d): Točnost=%.2f%% (%.2f%%) Točnost pri validaciji=%.2f%% (%.2f%%) Gubitak=%.2f (%.2f%%) Gubitak pri validaciji=%.2f (%.2f%%)' 
        % (begin_string, model_type, data_to_load, i + 1, 
        np.mean(accuracy[i])*100, np.std(accuracy[i])*100, 
        np.mean(val_acc[i])*100, np.std(val_acc[i])*100, 
        np.mean(loss[i]), np.std(loss[i])*100, 
        np.mean(val_loss[i]), np.std(val_loss[i])*100))
