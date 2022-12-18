from utils import MODEL_DATA_PATH, MY_MODEL_DATA_PATH, SEQ_MODEL_DATA_PATH, PATH_TO_NAME, final_acc_name, final_loss_name, acc_name, loss_name, predictions_name, history_name, final_history_name, results_name
import matplotlib.pyplot as plt
from custom_plots import hist_predicted, make_ROC_plots, make_PR_plots, merge_type_iteration

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
    plt.savefig("test1.png")
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
    plt.savefig("test2.png")
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
all_predictions = []
all_labels = []
for test_number in range(1, NUM_FOLDS_FIRST + 1):
    predictions, labels = read_one_prediction(MODEL_DATA_PATH, test_number, "weak", 1)
    all_predictions.append(predictions)
    all_labels.append(labels)
print(all_predictions, all_labels)
hist_predicted_multiple(
    MODEL_DATA_PATH,
    0,
    "weak",
    1,
    all_labels,
    all_predictions,
)