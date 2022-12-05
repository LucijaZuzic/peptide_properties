from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, auc, f1_score
import matplotlib.pyplot as plt
import numpy as np 

# Plot the history for training a model
def plt_model(MODEL_DATA_PATH, test_number, history, model_file_name):
    
    # Summarize history for accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2)
    plt.savefig(MODEL_DATA_PATH+str(test_number)+"_"+model_file_name+"_acc.png", bbox_inches='tight')
    plt.close()
    # Summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2)
    plt.savefig(MODEL_DATA_PATH+str(test_number)+"_"+model_file_name+"_loss.png", bbox_inches='tight')
    plt.close()
    
# Plot the history for training a model
def plt_model_final(MODEL_DATA_PATH, test_number, history, model_file_name):
    
    # Summarize history for accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Accuracy') 
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2)
    plt.savefig(MODEL_DATA_PATH+str(test_number)+"_"+model_file_name+"_acc.png", bbox_inches='tight')
    plt.close()
    # Summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'], label='Loss') 
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2)
    plt.savefig(MODEL_DATA_PATH+str(test_number)+"_"+model_file_name+"_loss.png", bbox_inches='tight')
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
    print(label+': Threshold=%f, F1=%.3f' % (thresholds[ix], fscore[ix]))
    plt.plot(recall[ix], precision[ix], 'o', markerfacecolor=pattern, markeredgecolor='k')

def make_PR_plots(MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions=[]):
    
    plt.figure()
    plt.title("Precision - Recall (PR) curve")

    if len(model_predictions) != 0:
        make_a_PR_plot(test_labels, model_predictions, 'r', 'PR curve for multiple properties model ')
         
	# Calculate the no skill line as the proportion of the positive class
    num_positive = 0
    for value in test_labels:
        if value == 1:
            num_positive += 1
    no_skill = num_positive / len(test_labels)

	# Plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], 'c', label='PR curve for random guessing')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2)
    plt.savefig(MODEL_DATA_PATH+str(test_number)+"_PR_curve_multiple_properties" + iteration + ".png", bbox_inches = 'tight')
        
    plt.close()

def make_a_ROC_plot(test_labels, model_predictions, pattern, label):

    plt.title("Receiver operating characteristic (ROC)")
    
    # Get false positive rate and true positive rate.
    fpr, tpr, thresholds = roc_curve(test_labels, model_predictions)

    # Plot ROC curve.
    plt.plot(fpr, tpr, pattern, label=label)

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))

    # Locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print(label+': Threshold=%f, Geometric mean=%.3f' % (thresholds[ix], gmeans[ix]))
    plt.plot(fpr[ix], tpr[ix], 'o', markerfacecolor=pattern, markeredgecolor='k')

def make_ROC_plots(MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions=[]):
    
    plt.figure()

    make_a_ROC_plot(test_labels, model_predictions, 'r', 'ROC curve for multiple properties model ')

    # Plot random guessing ROC curve.
    plt.plot([0, 1], [0, 1], 'c', label='ROC curve for random guessing')

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2)
    plt.savefig(MODEL_DATA_PATH+str(test_number)+"_ROC_curve_multiple_properties" + iteration + ".png", bbox_inches = 'tight')
    
    plt.close()

# Count correct predictions based on a custom threshold of probability
def my_accuracy_calculate(test_labels, model_predictions, threshold=0.5):

    score = 0

    model_predictions = convert_to_binary(model_predictions, threshold)

    for i in range(len(test_labels)):

        if (model_predictions[i] == test_labels[i]):
            score += 1
        
    return score / len(test_labels) * 100

def output_metrics(test_labels, model_predictions, threshold=0.5):
    # Get recall and precision.
    precision, recall, _ = precision_recall_curve(test_labels, model_predictions)

    # Convert probabilities to predictions
    model_predictions_binary = convert_to_binary(model_predictions, threshold)

    print("Multiple properties model: Accuracy: %.2f%% Area under ROC curve: %.4f Area under PR curve: %.4f F1: %.4f" 
                        % (
                        my_accuracy_calculate(test_labels, model_predictions, threshold), 
                        roc_auc_score(test_labels, model_predictions), 
                        auc(recall, precision), 
                        f1_score(test_labels, model_predictions_binary)))

def hist_predicted(MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions):

    # Create a histogram of the predicted probabilities only for the peptides that show self-assembly

    model_predictions_true = []
    for x in range(len(test_labels)):
        if test_labels[x] == 1.0:
            model_predictions_true.append(float(model_predictions[x]))

    plt.figure()
    plt.title("Multiple properties model\nHistogram of predicted self assembly probability\nfor peptides with self assembly")
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides")
    plt.hist(model_predictions_true, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    
    plt.savefig(MODEL_DATA_PATH+str(test_number)+"_"+"model_multiple_properties_histogram_SA" + iteration + ".png", bbox_inches = 'tight')
   
    # Create a histogram of the predicted probabilities only for the peptides that don't show self-assembly

    model_predictions_false = []
    for x in range(len(test_labels)):
        if test_labels[x] == 0.0:
            model_predictions_false.append(float(model_predictions[x]))

    plt.figure()
    plt.title("Multiple properties model\nHistogram of predicted self assembly probability\nfor peptides without self assembly")
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides")
    plt.hist(model_predictions_false, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    
    plt.savefig(MODEL_DATA_PATH+str(test_number)+"_"+"model_multiple_properties_histogram_NSA" + iteration + ".png", bbox_inches = 'tight')

def decorate_stats(history, params_nr='', fold_nr=''):
    accuracy =history.history['accuracy'] 
    loss = history.history['loss'] 
    val_loss = history.history['val_loss']  
    val_acc = history.history['val_accuracy']
    
    accuracy_max = np.max(accuracy)
    val_acc_max = np.max(val_acc) 
    loss_min = np.min(loss) 
    val_loss_min = np.min(val_loss)

    begin_string =  "Multiple properties model (params " + str(params_nr) + ", fold " + str(fold_nr) + ")" 
    if params_nr=='' and fold_nr=='':
        begin_string = "Best multiple properties model"

    print('%s: Maximum accuracy=%.2f%% Maximum validation accuracy=%.2f%% Minimal loss=%.2f%% Minimal validation loss=%.2f%%' 
    % (begin_string, accuracy_max*100, val_acc_max*100, loss_min*100, val_loss_min*100))
    print('%s: Accuracy=%.2f%% (%.2f%%) Validation accuracy=%.2f%% (%.2f%%) Loss=%.2f%% (%.2f%%) Validation loss=%.2f%% (%.2f%%)' 
    % (begin_string, 
    np.mean(accuracy)*100, np.std(accuracy)*100, 
    np.mean(val_acc)*100, np.std(val_acc)*100, 
    np.mean(loss)*100, np.std(loss)*100, 
    np.mean(val_loss)*100, np.std(val_loss)*100))
    
def decorate_stats_final(history, params_nr='', fold_nr=''):
    accuracy =history.history['accuracy'] 
    loss = history.history['loss'] 
    
    accuracy_max = np.max(accuracy)
    loss_min = np.min(loss) 

    begin_string =  "Multiple properties model (params " + str(params_nr) + ", fold " + str(fold_nr) + ")" 
    if params_nr=='' and fold_nr=='':
        begin_string = "Best multiple properties model"

    print('%s: Maximum accuracy=%.2f%% Minimal loss=%.2f%%' 
    % (begin_string, accuracy_max*100, loss_min*100))
    print('%s: Accuracy=%.2f%% (%.2f%%) Loss=%.2f%% (%.2f%%)' 
    % (begin_string, 
    np.mean(accuracy)*100, np.std(accuracy)*100, 
    np.mean(loss)*100, np.std(loss)*100))
    
def decorate_stats_avg(accuracy, val_acc, loss, val_loss, params_nr=''):  
    
    accuracy_max = np.max(accuracy)
    val_acc_max = np.max(val_acc) 
    loss_min = np.min(loss) 
    val_loss_min = np.min(val_loss)

    print('Model average for multiple properties model (params %d): Maximum accuracy=%.2f%% (%.2f%%) Maximum validation accuracy=%.2f%% (%.2f%%) Minimal loss=%.2f%% (%.2f%%) Minimal validation loss=%.2f%% (%.2f%%)' 
    % (params_nr, np.mean(accuracy_max)*100, np.std(accuracy_max)*100, np.mean(val_acc_max)*100, np.std(val_acc_max)*100, np.mean(loss_min), np.std(loss_min)*100, np.mean(val_loss_min), np.std(val_loss_min)*100))
    print('Model average for multiple properties model (params %d): Accuracy=%.2f%% (%.2f%%) Validation accuracy=%.2f%% (%.2f%%) Loss=%.2f%% (%.2f%%) Validation loss=%.2f%% (%.2f%%)' 
    % (params_nr, np.mean(accuracy)*100, np.std(accuracy)*100, np.mean(val_acc)*100, np.std(val_acc)*100, np.mean(loss), np.std(loss)*100, np.mean(val_loss), np.std(val_loss)*100))