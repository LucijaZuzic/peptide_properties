from utils import MODEL_DATA_PATH, MY_MODEL_DATA_PATH, SEQ_MODEL_DATA_PATH, PATH_TO_NAME, PATH_TO_EXTENSION, final_acc_name, final_loss_name, acc_name, loss_name, predictions_name, history_name, final_history_name, results_name
import matplotlib.pyplot as plt
from custom_plots import hist_predicted, make_ROC_plots, make_PR_plots, merge_type_iteration
import numpy as np

NUM_FOLDS_FIRST = 5
NUM_FOLDS_SECOND = 5
MAX_ITERATION = 5
MAX_PARAMS = 9
MIN_PARAMS = 3

def hist_predicted_multiple(
    MODEL_DATA_PATH,
    test_number,
    final_model_type,
    iteration,
    test_labels,
    model_predictions,
    SAVE
):
    # Create a histogram of the predicted probabilities only for the peptides that show self-assembly
    plot_label = []
    for model_num in range(1, 1 + len(model_predictions)):
        plot_label.append("Test " + str(model_num))

    model_predictions_true = []
    for model_num in range(len(model_predictions)):
        model_predictions_true.append([])
        for x in range(len(test_labels[model_num])):
            if test_labels[model_num][x] == 1.0:
                model_predictions_true[-1].append(float(model_predictions[model_num][x]))

    plt.figure()
    plt.title(
        merge_type_iteration(MODEL_DATA_PATH, final_model_type, iteration, test_number).replace("Test 0 Weak 1", "All tests")
        + "\nHistogram of predicted self assembly probability\nfor peptides with self assembly"
    )
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides")
    plt.hist(
        model_predictions_true, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], label=plot_label
    ) 
    plt.legend()
    plt.savefig(SAVE + "_SA.png")
    plt.close()

    # Create a histogram of the predicted probabilities only for the peptides that don't show self-assembly

    model_predictions_false = []
    for model_num in range(len(model_predictions)):
        model_predictions_false.append([])
        for x in range(len(test_labels[model_num])):
            if test_labels[model_num][x] == 0.0:
                model_predictions_false[-1].append(float(model_predictions[model_num][x]))

    plt.figure()
    plt.title(
        merge_type_iteration(MODEL_DATA_PATH, final_model_type, iteration, test_number).replace("Test 0 Weak 1", "All tests")
        + "\nHistogram of predicted self assembly probability\nfor peptides without self assembly"
    )
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides")
    plt.hist(
        model_predictions_false,
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], label=plot_label
    ) 
    plt.legend()
    plt.savefig(SAVE + "_NSA.png")
    plt.close() 

# Plot the history for training a model
def plt_model_no_history(some_path, test_number, fold_nr, params_nr):
    acc, val_acc, loss, val_loss = read_one_history(some_path, test_number, params_nr, fold_nr)
    # Summarize history for accuracy
    plt.figure()
    plt.plot(acc, label="Accuracy")
    plt.plot(val_acc, label="Validation accuracy")
    plt.title(
        PATH_TO_NAME[some_path]
        + " Params "
        + str(params_nr)
        + " Test "
        + str(test_number)
        + " Accuracy"
    )
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig(
        acc_name(some_path, test_number, params_nr, fold_nr), bbox_inches="tight"
    )
    plt.close()
    # Summarize history for loss
    plt.figure()
    plt.plot(loss, label="Loss")
    plt.plot(val_loss, label="Validation loss")
    plt.title(
        PATH_TO_NAME[some_path]
        + " Params "
        + str(params_nr)
        + " Test "
        + str(test_number)
        + " Loss"
    )
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig(
        loss_name(some_path, test_number, params_nr, fold_nr), bbox_inches="tight"
    )
    plt.close()


# Plot the history for training a model
def plt_model_final_no_history(iteration, some_path, test_number):
    acc, loss = read_one_final_history(some_path, test_number, iteration)
    # Summarize history for accuracy
    plt.figure()
    plt.plot(acc, label="Accuracy")
    plt.title(
        PATH_TO_NAME[some_path] + " " + merge_type_iteration("weak", iteration) + " Accuracy"
    )
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig(
        final_acc_name(some_path, test_number, iteration), bbox_inches="tight"
    )
    plt.close()
    # Summarize history for loss
    plt.figure()
    plt.plot(loss, label="Loss")
    plt.title(PATH_TO_NAME[some_path] + " " + merge_type_iteration("weak", iteration) + " Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig(
        final_loss_name(some_path, test_number, iteration), bbox_inches="tight"
    )
    plt.close()

def fix_error(some_path, test_number, first, second):     
    lines = read_one_results(some_path, test_number)
    count_error = 0
    for line_num in range(len(lines)):
        count_error += lines[line_num].count(first)
        lines[line_num] = lines[line_num].replace(first, second)
    print(count_error)
    return lines

def read_one_results(some_path, test_number):       
    file = open(results_name(some_path, test_number), "r") 
    lines = file.readlines() 
    file.close()
    return lines

def rewrite(some_path, test_number, first, second):       
    new_lines = fix_error(some_path, test_number, first, second)
    write_str = ""
    for line in new_lines:
        write_str += line
    file = open(results_name(some_path, test_number), "w") 
    file.write(write_str)
    file.close()

def read_one_history(some_path, test_number, params_nr, fold_nr):
    acc_path, val_acc_path, loss_path, val_loss_path = history_name(some_path, test_number, params_nr, fold_nr)
    file = open(acc_path, "r")
    lines = file.readlines()
    acc = eval(lines[0]) 
    file.close()
    file = open(val_acc_path, "r")
    lines = file.readlines()
    val_acc = eval(lines[0]) 
    file.close()
    file = open(loss_path, "r")
    lines = file.readlines()
    loss = eval(lines[0]) 
    file.close()
    file = open(val_loss_path, "r")
    lines = file.readlines()
    val_loss= eval(lines[0]) 
    file.close()
    return acc, val_acc, loss, val_loss  

def read_one_final_history(some_path, test_number, iteration):
    acc_path, loss_path = final_history_name(some_path, test_number, iteration)
    file = open(acc_path, "r")
    lines = file.readlines()
    acc = eval(lines[0]) 
    file.close() 
    file = open(loss_path, "r")
    lines = file.readlines()
    loss = eval(lines[0]) 
    file.close() 
    return acc, loss  

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

def fix_plots(path, test_number, final_model_type, iteration):
    model_predictions, test_labels  = read_one_prediction(path, test_number)

    #Plot ROC curves for all models
    make_ROC_plots(path, test_number, final_model_type, iteration, test_labels, model_predictions)
    
    #Plot PR curves for all models
    make_PR_plots(path, test_number, final_model_type, iteration, test_labels, model_predictions)
    
    # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
    hist_predicted(path, test_number, final_model_type, iteration, test_labels, model_predictions) 
 
def interesting_lines(some_path, test_number, find_str):     
    lines = read_one_results(some_path, test_number) 
    new_lines = []
    for line_num in range(len(lines)):
        if lines[line_num].count(find_str) > 0 and (lines[line_num].count("Weak 1") > 0 or lines[line_num].count("Best params") > 0):   
            new_lines.append(lines[line_num])
    return new_lines

def value_for_test_dev(some_path, test_number, param, capture, capture_dev, keyword, divide):
    line = interesting_lines(some_path, test_number, keyword)[0]
    index = line.find(param) + len(param)
    number = line[index:index + capture].replace(" ", "")
    dev = line[index + capture + 3:index + capture + 3 + capture_dev].replace(" ", "")
    number = float(number) / divide
    dev = float(dev) / divide
    return number, dev
 
def add_line_dev(add, line, alist, alist_dev, num_and_dev, add_percent):
    num = num_and_dev[0]
    dev = num_and_dev[1]
    alist.append(num) 
    alist_dev.append(dev)  
    if add:
        if add_percent:
            line = "%s & %.2f%% (%.2f%%)" %(line, num, dev)
        else:
            line = "%s & %.2f (%.2f)" %(line, num, dev)
    return line, alist, alist_dev

def value_for_test(some_path, test_number, param, capture, keyword, is_int, divide):
    line = interesting_lines(some_path, test_number, keyword)[0]
    index = line.find(param) + len(param)
    number = line[index:index + capture].replace(" ", "")
    if is_int:
        number = int(number)  
    else:
        number = float(number) / divide
    return number
 
def add_line(add, line, alist, num, add_percent, is_int):
    alist.append(num) 
    if add:
        if is_int:
            line = "%s & %d" %(line, num)
        else:
            line = "%s & %.2f" %(line, num)
        if add_percent:
            line += "%"
    return line, alist

def make_a_table(some_path, include_avg, filter = None):
    header = "Value"

    for test_number in range(1, NUM_FOLDS_FIRST + 1):  
        if filter == None or test_number == filter:
            header += " & Test " + str(test_number)

    line_num_cells = "Output dimension in final layer"
    line_kernel_size = "Kernel size for convolutional layer"

    max_acc_line = "Maximal accuracy"
    min_loss_line = "Minimal loss"
    avg_acc_line = "Average accuracy (standard deviation)"
    avg_loss_line = "Average loss (standard deviation)"
    ROC_thr_line = "ROC self-assembly threshold"
    ROC_gmean_line = "ROC geometric mean"
    PR_thr_line = "PR self-assembly threshold"
    F1_train_line = "PR F1"

    acc_line = "Accuracy"
    ROC_AUC_line = "Area under ROC curve"
    PR_line = "Area under PR curve"
    F1_test_line = "PR F1"

    num_cells_list = []
    kernel_size_list = []

    max_acc_list = []
    min_loss_list = []
    avg_acc_list = []
    avg_acc_dev_list = []
    avg_loss_list = []
    avg_loss_dev_list = []
    ROC_thr_list = []
    ROC_gmean_list = []
    PR_thr_list = []
    F1_train_list = []

    acc_list = []
    ROC_AUC_list = []
    PR_list = []
    F1_test_list = []

    max_index = 0
    max_acc = 0

    for test_number in range(1, NUM_FOLDS_FIRST + 1):  
        add = False
        if filter == None or test_number == filter:
            add = True
        if some_path != MY_MODEL_DATA_PATH:    
            line_num_cells, num_cells_list = add_line(add, line_num_cells, num_cells_list, value_for_test(some_path, test_number, "num_cells: ", 3, "Best params", True, 1), False, True)
            line_kernel_size, kernel_size_list = add_line(add, line_kernel_size, kernel_size_list, value_for_test(some_path, test_number, "kernel_size: ", 1, "Best params", True, 1), False, True) 
        else:
            line_num_cells, num_cells_list = add_line(add, line_num_cells, num_cells_list, value_for_test(some_path, test_number, "dense: ", 3, "Best params", True, 1), False, True)

        max_acc_line, max_acc_list = add_line(add, max_acc_line, max_acc_list, value_for_test(some_path, test_number, "Maximum accuracy = ", 5, "Maximum accuracy = ", False, 1), True, False) 
        min_loss_line, min_loss_list = add_line(add, min_loss_line, min_loss_list, value_for_test(some_path, test_number, "% Minimal loss = ", 4, "% Minimal loss = ", False, 1), False, False) 
        avg_acc_line, avg_acc_list, avg_acc_dev_list = add_line_dev(add, avg_acc_line, avg_acc_list, avg_acc_dev_list, value_for_test_dev(some_path, test_number, ": Accuracy = ", 5, 4, ": Accuracy = ", 1), True)
        avg_loss_line, avg_loss_list, avg_loss_dev_list = add_line_dev(add, avg_loss_line, avg_loss_list, avg_loss_dev_list, value_for_test_dev(some_path, test_number, "Loss = ", 4, 4, "Loss = ", 1), False)
        ROC_thr_line, ROC_thr_list = add_line(add, ROC_thr_line, ROC_thr_list, value_for_test(some_path, test_number, "ROC curve: Threshold = ", 8, "ROC curve: Threshold = ", False, 1), False, False) 
        ROC_gmean_line, ROC_gmean_list = add_line(add, ROC_gmean_line, ROC_gmean_list, value_for_test(some_path, test_number, ", Geometric mean = ", 5, ", Geometric mean = ", False, 1), False, False) 
        PR_thr_line, PR_thr_list = add_line(add, PR_thr_line, PR_thr_list, value_for_test(some_path, test_number, "PR curve: Threshold = ", 8, "PR curve: Threshold = ", False, 1), False, False) 
        F1_train_line, F1_train_list = add_line(add, F1_train_line, F1_train_list, value_for_test(some_path, test_number, ", F1 = ", 5, ", F1 = ", False, 1), False, False) 
        
        acc_line, acc_list = add_line(add, acc_line, acc_list, value_for_test(some_path, test_number, "Accuracy: ", 5, "Accuracy: ", False, 1), False, False) 
        ROC_AUC_line, ROC_AUC_list = add_line(add, ROC_AUC_line, ROC_AUC_list, value_for_test(some_path, test_number, "Area under ROC curve: ", 6, "Area under ROC curve: ", False, 1), False, False) 
        PR_line, PR_list = add_line(add, PR_line, PR_list, value_for_test(some_path, test_number, "Area under PR curve: ", 6, "Area under PR curve: ", False, 1), False, False) 
        F1_test_line, F1_test_list = add_line(add, F1_test_line, F1_test_list, value_for_test(some_path, test_number, "F1: ", 6, "F1: ", False, 1), False, False) 
    
        if acc_list[-1] > max_acc:
            max_acc = acc_list[-1]
            max_index = test_number
    
    if include_avg: 
        header += " & Average \\\\ \n"
        header += ("%s & %.2f \\\\ \n" % (line_num_cells, np.mean(num_cells_list)))
        if some_path != MY_MODEL_DATA_PATH: 
            header += ("%s & %.2f \\\\ \n" % (line_kernel_size, np.mean(kernel_size_list))) 

        header += ("%s & %.2f%% \\\\ \n" % (max_acc_line, np.mean(max_acc_list)))
        header += ("%s & %.2f \\\\ \n" % (min_loss_line, np.mean(min_loss_list)))  
        header += ("%s & %.2f%% (%.2f%%) \\\\ \n" % (avg_acc_line, np.mean(avg_acc_list), np.mean(avg_acc_dev_list)))
        header += ("%s & %.2f (%.2f) \\\\ \n" % (avg_loss_line, np.mean(avg_loss_list), np.mean(avg_loss_dev_list)))
        header += ("%s & %.2f \\\\ \n" % (ROC_thr_line, np.mean(ROC_thr_list)))
        header += ("%s & %.2f \\\\ \n" % (ROC_gmean_line, np.mean(ROC_gmean_list))) 
        header += ("%s & %.2f \\\\ \n" % (PR_thr_line, np.mean(PR_thr_list)))
        header += ("%s & %.2f \\\\ \n" % (F1_train_line, np.mean(F1_train_list))) 

        header += ("%s & %.2f \\\\ \n" % (acc_line, np.mean(acc_list)))
        header += ("%s & %.2f \\\\ \n" % (ROC_AUC_line, np.mean(ROC_AUC_list))) 
        header += ("%s & %.2f \\\\ \n" % (PR_line, np.mean(PR_list)))
        header += ("%s & %.2f \\\\ \n" % (F1_test_line, np.mean(F1_test_list))) 
    else:
        header += " \\\\ \n"
        header += ("%s \\\\ \n" % (line_num_cells))
        if some_path != MY_MODEL_DATA_PATH: 
            header += ("%s \\\\ \n" % (line_kernel_size)) 

        header += ("%s \\\\ \n" % (max_acc_line))
        header += ("%s \\\\ \n" % (min_loss_line))
        header += ("%s \\\\ \n" % (avg_acc_line))
        header += ("%s \\\\ \n" % (avg_loss_line))
        header += ("%s \\\\ \n" % (ROC_thr_line))
        header += ("%s \\\\ \n" % (ROC_gmean_line)) 
        header += ("%s \\\\ \n" % (PR_thr_line))
        header += ("%s \\\\ \n" % (F1_train_line))

        header += ("%s \\\\ \n" % (acc_line))
        header += ("%s \\\\ \n" % (ROC_AUC_line))
        header += ("%s \\\\ \n" % (PR_line))
        header += ("%s \\\\ \n" % (F1_test_line)) 

    if filter == None:
        return header, make_a_table(some_path, include_avg, max_index)
    return header

def merge_all(paths, include_avg, filter = None):
    header = "Value"

    for some_path in paths:
        for test_number in range(1, NUM_FOLDS_FIRST + 1):  
            if filter == None or test_number == filter:
                header += " & " + PATH_TO_NAME[some_path] + " Test " + str(test_number)
        if include_avg and filter == None:
            header += " & " + PATH_TO_NAME[some_path] + " average"  


    line_num_cells = "Output dimension in final layer"
    line_kernel_size = "Kernel size for convolutional layer"

    max_acc_line = "Maximal accuracy"
    min_loss_line = "Minimal loss"
    avg_acc_line = "Average accuracy (standard deviation)"
    avg_loss_line = "Average loss (standard deviation)"
    ROC_thr_line = "ROC self-assembly threshold"
    ROC_gmean_line = "ROC geometric mean"
    PR_thr_line = "PR self-assembly threshold"
    F1_train_line = "PR F1"

    acc_line = "Accuracy"
    ROC_AUC_line = "Area under ROC curve"
    PR_line = "Area under PR curve"
    F1_test_line = "PR F1"

    num_cells_list = []
    kernel_size_list = []

    max_acc_list = []
    min_loss_list = []
    avg_acc_list = []
    avg_acc_dev_list = []
    avg_loss_list = []
    avg_loss_dev_list = []
    ROC_thr_list = []
    ROC_gmean_list = []
    PR_thr_list = []
    F1_train_list = []

    acc_list = []
    ROC_AUC_list = []
    PR_list = []
    F1_test_list = []

    for some_path in paths:

        max_index = 0
        max_acc = 0 
        for test_number in range(1, NUM_FOLDS_FIRST + 1):  
            add = False
            if filter == None or test_number == filter:
                add = True 
            if some_path != MY_MODEL_DATA_PATH:    
                line_num_cells, num_cells_list = add_line(add, line_num_cells, num_cells_list, value_for_test(some_path, test_number, "num_cells: ", 3, "Best params", True, 1), False, True)
                line_kernel_size, kernel_size_list = add_line(add, line_kernel_size, kernel_size_list, value_for_test(some_path, test_number, "kernel_size: ", 1, "Best params", True, 1), False, True) 
            else:
                line_num_cells, num_cells_list = add_line(add, line_num_cells, num_cells_list, value_for_test(some_path, test_number, "dense: ", 3, "Best params", True, 1), False, True)
                if add:    
                    line_kernel_size += " & "

            max_acc_line, max_acc_list = add_line(add, max_acc_line, max_acc_list, value_for_test(some_path, test_number, "Maximum accuracy = ", 5, "Maximum accuracy = ", False, 1), True, False) 
            min_loss_line, min_loss_list = add_line(add, min_loss_line, min_loss_list, value_for_test(some_path, test_number, "% Minimal loss = ", 4, "% Minimal loss = ", False, 1), False, False) 
            avg_acc_line, avg_acc_list, avg_acc_dev_list = add_line_dev(add, avg_acc_line, avg_acc_list, avg_acc_dev_list, value_for_test_dev(some_path, test_number, ": Accuracy = ", 5, 4, ": Accuracy = ", 1), True)
            avg_loss_line, avg_loss_list, avg_loss_dev_list = add_line_dev(add, avg_loss_line, avg_loss_list, avg_loss_dev_list, value_for_test_dev(some_path, test_number, "Loss = ", 4, 4, "Loss = ", 1), False)
            ROC_thr_line, ROC_thr_list = add_line(add, ROC_thr_line, ROC_thr_list, value_for_test(some_path, test_number, "ROC curve: Threshold = ", 8, "ROC curve: Threshold = ", False, 1), False, False) 
            ROC_gmean_line, ROC_gmean_list = add_line(add, ROC_gmean_line, ROC_gmean_list, value_for_test(some_path, test_number, ", Geometric mean = ", 5, ", Geometric mean = ", False, 1), False, False) 
            PR_thr_line, PR_thr_list = add_line(add, PR_thr_line, PR_thr_list, value_for_test(some_path, test_number, "PR curve: Threshold = ", 8, "PR curve: Threshold = ", False, 1), False, False) 
            F1_train_line, F1_train_list = add_line(add, F1_train_line, F1_train_list, value_for_test(some_path, test_number, ", F1 = ", 5, ", F1 = ", False, 1), False, False) 
            
            acc_line, acc_list = add_line(add, acc_line, acc_list, value_for_test(some_path, test_number, "Accuracy: ", 5, "Accuracy: ", False, 1), False, False) 
            ROC_AUC_line, ROC_AUC_list = add_line(add, ROC_AUC_line, ROC_AUC_list, value_for_test(some_path, test_number, "Area under ROC curve: ", 6, "Area under ROC curve: ", False, 1), False, False) 
            PR_line, PR_list = add_line(add, PR_line, PR_list, value_for_test(some_path, test_number, "Area under PR curve: ", 6, "Area under PR curve: ", False, 1), False, False) 
            F1_test_line, F1_test_list = add_line(add, F1_test_line, F1_test_list, value_for_test(some_path, test_number, "F1: ", 6, "F1: ", False, 1), False, False) 
        
            if acc_list[-1] > max_acc:
                max_acc = acc_list[-1]
                max_index = test_number

        if include_avg and filter == None: 
            change_size = NUM_FOLDS_FIRST
            line_num_cells = ("%s & %.2f" % (line_num_cells, np.mean(num_cells_list[-change_size:-1])))
            if some_path != MY_MODEL_DATA_PATH:   
                line_kernel_size = ("%s & %.2f" % (line_kernel_size, np.mean(kernel_size_list[-change_size:-1])))
            else: 
                line_kernel_size = ("%s & " % (line_kernel_size))

            max_acc_line = ("%s & %.2f%%" % (max_acc_line, np.mean(max_acc_list[-change_size:-1])))
            min_loss_line = ("%s & %.2f" % (min_loss_line, np.mean(min_loss_list[-change_size:-1])))  
            avg_acc_line = ("%s & %.2f%% (%.2f%%)" % (avg_acc_line, np.mean(avg_acc_list), np.mean(avg_acc_dev_list[-change_size:-1])))
            avg_loss_line = ("%s & %.2f (%.2f)" % (avg_loss_line, np.mean(avg_loss_list), np.mean(avg_loss_dev_list[-change_size:-1])))
            ROC_thr_line = ("%s & %.2f" % (ROC_thr_line, np.mean(ROC_thr_list[-change_size:-1])))
            ROC_gmean_line = ("%s & %.2f" % (ROC_gmean_line, np.mean(ROC_gmean_list[-change_size:-1]))) 
            PR_thr_line = ("%s & %.2f" % (PR_thr_line, np.mean(PR_thr_list[-change_size:-1])))
            F1_train_line = ("%s & %.2f" % (F1_train_line, np.mean(F1_train_list[-change_size:-1]))) 

            acc_line = ("%s & %.2f" % (acc_line, np.mean(acc_list[-change_size:-1])))
            ROC_AUC_line = ("%s & %.2f" % (ROC_AUC_line, np.mean(ROC_AUC_list[-change_size:-1]))) 
            PR_line = ("%s & %.2f" % (PR_line, np.mean(PR_list[-change_size:-1])))
            F1_test_line = ("%s & %.2f" % (F1_test_line, np.mean(F1_test_list[-change_size:-1]))) 
          
    if include_avg: 
        header += " & Average \\\\ \n"
        header += ("%s & %.2f \\\\ \n" % (line_num_cells, np.mean(num_cells_list)))
        header += ("%s & %.2f \\\\ \n" % (line_kernel_size, np.mean(kernel_size_list))) 

        header += ("%s & %.2f%% \\\\ \n" % (max_acc_line, np.mean(max_acc_list)))
        header += ("%s & %.2f \\\\ \n" % (min_loss_line, np.mean(min_loss_list)))  
        header += ("%s & %.2f%% (%.2f%%) \\\\ \n" % (avg_acc_line, np.mean(avg_acc_list), np.mean(avg_acc_dev_list)))
        header += ("%s & %.2f (%.2f) \\\\ \n" % (avg_loss_line, np.mean(avg_loss_list), np.mean(avg_loss_dev_list)))
        header += ("%s & %.2f \\\\ \n" % (ROC_thr_line, np.mean(ROC_thr_list)))
        header += ("%s & %.2f \\\\ \n" % (ROC_gmean_line, np.mean(ROC_gmean_list))) 
        header += ("%s & %.2f \\\\ \n" % (PR_thr_line, np.mean(PR_thr_list)))
        header += ("%s & %.2f \\\\ \n" % (F1_train_line, np.mean(F1_train_list))) 

        header += ("%s & %.2f \\\\ \n" % (acc_line, np.mean(acc_list)))
        header += ("%s & %.2f \\\\ \n" % (ROC_AUC_line, np.mean(ROC_AUC_list))) 
        header += ("%s & %.2f \\\\ \n" % (PR_line, np.mean(PR_list)))
        header += ("%s & %.2f \\\\ \n" % (F1_test_line, np.mean(F1_test_list))) 
    else:
        header += " \\\\ \n"
        header += ("%s \\\\ \n" % (line_num_cells))
        header += ("%s \\\\ \n" % (line_kernel_size)) 

        header += ("%s \\\\ \n" % (max_acc_line))
        header += ("%s \\\\ \n" % (min_loss_line))
        header += ("%s \\\\ \n" % (avg_acc_line))
        header += ("%s \\\\ \n" % (avg_loss_line))
        header += ("%s \\\\ \n" % (ROC_thr_line))
        header += ("%s \\\\ \n" % (ROC_gmean_line)) 
        header += ("%s \\\\ \n" % (PR_thr_line))
        header += ("%s \\\\ \n" % (F1_train_line))

        header += ("%s \\\\ \n" % (acc_line))
        header += ("%s \\\\ \n" % (ROC_AUC_line))
        header += ("%s \\\\ \n" % (PR_line))
        header += ("%s \\\\ \n" % (F1_test_line)) 

    if filter == None:
        return header, merge_all(paths, include_avg, max_index)
    return header

#all_predictions, all_labels = read_all_model_predictions(SEQ_MODEL_DATA_PATH, 1, NUM_FOLDS_FIRST)

'''
for test_number in range(-1, NUM_FOLDS_FIRST + 1):
    rewrite(MODEL_DATA_PATH, test_number, "Weak", "Test " + str(test_number) + " Weak")
    rewrite(SEQ_MODEL_DATA_PATH, test_number, "Weak", "Test " + str(test_number) + " Weak")
    rewrite(MY_MODEL_DATA_PATH, test_number, "Weak", "Test " + str(test_number) + " Weak")

    fix_error(MODEL_DATA_PATH, test_number, "Weak", "Test " + str(test_number) + " Weak")
    fix_error(SEQ_MODEL_DATA_PATH, test_number, "Weak", "Test " + str(test_number) + " Weak")
    fix_error(MY_MODEL_DATA_PATH, test_number, "Weak", "Test " + str(test_number) + " Weak")

    rewrite(MODEL_DATA_PATH, test_number, "Iteration", "Test " + str(test_number) + " Iteration")
    rewrite(SEQ_MODEL_DATA_PATH, test_number, "Iteration", "Test " + str(test_number) + " Iteration")
    rewrite(MY_MODEL_DATA_PATH, test_number, "Iteration", "Test " + str(test_number) + " Iteration") 

    fix_error(MODEL_DATA_PATH, test_number, "Iteration", "Test " + str(test_number) + " Iteration")
    fix_error(SEQ_MODEL_DATA_PATH, test_number, "Iteration", "Test " + str(test_number) + " Iteration")
    fix_error(MY_MODEL_DATA_PATH, test_number, "Iteration", "Test " + str(test_number) + " Iteration")

    for iteration in range(1, MAX_ITERATION + 1):
        plt_model_final_no_history(iteration, MODEL_DATA_PATH, test_number)
        plt_model_final_no_history(iteration, SEQ_MODEL_DATA_PATH, test_number)
        plt_model_final_no_history(iteration, MY_MODEL_DATA_PATH, test_number)
    for fold_nr in range(1, NUM_FOLDS_SECOND + 1):
        for params_nr in range(1, MAX_PARAMS + 1):
            plt_model_no_history(MODEL_DATA_PATH, test_number, fold_nr, params_nr)
            plt_model_no_history(SEQ_MODEL_DATA_PATH, test_number, fold_nr, params_nr)
        for params_nr in range(1, MIN_PARAMS + 1):
            plt_model_no_history(MY_MODEL_DATA_PATH, test_number, fold_nr, params_nr)
    fix_plots(MODEL_DATA_PATH, test_number, 'weak', iteration)
    fix_plots(SEQ_MODEL_DATA_PATH, test_number, 'weak', iteration)
    fix_plots(MY_MODEL_DATA_PATH, test_number, 'weak', iteration)
    fix_plots(MODEL_DATA_PATH, test_number, 'iteration', iteration)
    fix_plots(SEQ_MODEL_DATA_PATH, test_number, 'iteration', iteration)
    fix_plots(MY_MODEL_DATA_PATH, test_number, 'iteration', iteration)

'''
def make_cumulative_hist(path):
    all_predictions = []
    all_labels = []
    for test_number in range(1, NUM_FOLDS_FIRST + 1):
        predictions, labels = read_one_prediction(path, test_number, "weak", 1)
        all_predictions.append(predictions)
        all_labels.append(labels)
    print(all_predictions, all_labels)
    hist_predicted_multiple(
        path,
        0,
        "weak",
        1,
        all_labels,
        all_predictions,
        path + PATH_TO_EXTENSION[path] + "_hist_all"
    )
make_cumulative_hist(MODEL_DATA_PATH)
make_cumulative_hist(MY_MODEL_DATA_PATH)
make_cumulative_hist(SEQ_MODEL_DATA_PATH)
res = make_a_table(MODEL_DATA_PATH, True)
print(res[0])
print(res[1])
res = make_a_table(SEQ_MODEL_DATA_PATH, True)
print(res[0])
print(res[1])
res = make_a_table(MY_MODEL_DATA_PATH, True)
print(res[0])
print(res[1])
res = merge_all([MODEL_DATA_PATH, SEQ_MODEL_DATA_PATH, MY_MODEL_DATA_PATH], True)
print(res[0])
print(res[1])