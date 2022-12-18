from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    auc,
    f1_score,
)
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    hist_names,
    ROC_name,
    PR_name,
    PATH_TO_NAME,
    results_name,
    final_history_name,
    history_name,
    predictions_name,
    final_acc_name,
    final_loss_name,
    loss_name,
    acc_name,
)


def merge_type_test_number(some_path, test_number):
    return "Model " + PATH_TO_NAME[some_path] + " Test " + str(test_number)


def merge_type_iteration(some_path, final_model_type, iteration, test_number):
    new_iteration = (
        merge_type_test_number(some_path, test_number)
        + " "
        + final_model_type.replace("weak", "Weak")
    )
    new_iteration = new_iteration.replace("iteration", "Iteration")
    new_iteration += " " + str(iteration)
    return new_iteration


def merge_type_params(some_path, fold_nr, params_nr, test_number):
    return (
        merge_type_test_number(some_path, test_number)
        + " Params "
        + str(params_nr)
        + " Fold "
        + str(fold_nr)
    )


# Plot the history for training a model
def plt_model(params_nr, some_path, test_number, history, fold_nr):
    # Summarize history for accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.title(
        merge_type_params(some_path, fold_nr, params_nr, test_number) + " Accuracy"
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
    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title(merge_type_params(some_path, fold_nr, params_nr, test_number) + " Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig(
        loss_name(some_path, test_number, params_nr, fold_nr), bbox_inches="tight"
    )
    plt.close()


# Plot the history for training a model
def plt_model_final(iteration, some_path, test_number, history):
    # Summarize history for accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="Accuracy")
    plt.title(
        merge_type_iteration(some_path, "weak", iteration, test_number) + " Accuracy"
    )
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig(final_acc_name(some_path, test_number, iteration), bbox_inches="tight")
    plt.close()
    # Summarize history for loss
    plt.figure()
    plt.plot(history.history["loss"], label="Loss")
    plt.title(merge_type_iteration(some_path, "weak", iteration, test_number) + " Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig(final_loss_name(some_path, test_number, iteration), bbox_inches="tight")
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


def make_a_PR_plot(
    some_path,
    test_number,
    final_model_type,
    iteration,
    test_labels,
    model_predictions,
    pattern,
    label,
):
    # Get recall and precision.
    precision, recall, thresholds = precision_recall_curve(
        test_labels, model_predictions
    )

    # Plot PR curve.
    plt.plot(recall, precision, pattern, label=label)

    # Calculate the F1 score for each threshold
    fscore = []
    for i in range(len(precision)):
        fscore.append(
            weird_division(2 * precision[i] * recall[i], precision[i] + recall[i])
        )

    # Locate the index of the largest F1 score
    ix = np.argmax(fscore)
    other_output = open(results_name(some_path, test_number), "a", encoding="utf-8")
    other_output.write(
        merge_type_iteration(some_path, final_model_type, iteration, test_number)
        + " "
        + label
        + ": Threshold = %f, F1 = %.3f" % (thresholds[ix], fscore[ix])
    )
    other_output.write("\n")
    other_output.close()
    plt.plot(
        recall[ix], precision[ix], "o", markerfacecolor=pattern, markeredgecolor="k"
    )


def make_PR_plots(
    some_path,
    test_number,
    final_model_type,
    iteration,
    test_labels,
    model_predictions=[],
):
    plt.figure()
    plt.title(
        merge_type_iteration(some_path, final_model_type, iteration, test_number)
        + "\nPrecision - Recall (PR) curve"
    )

    if len(model_predictions) != 0:
        make_a_PR_plot(
            some_path,
            test_number,
            final_model_type,
            iteration,
            test_labels,
            model_predictions,
            "r",
            "PR curve",
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

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig(
        PR_name(some_path, test_number, final_model_type, iteration),
        bbox_inches="tight",
    )

    plt.close()


def make_a_ROC_plot(
    some_path,
    test_number,
    final_model_type,
    iteration,
    test_labels,
    model_predictions,
    pattern,
    label,
):
    # Get false positive rate and true positive rate.
    fpr, tpr, thresholds = roc_curve(test_labels, model_predictions)

    # Plot ROC curve.
    plt.plot(fpr, tpr, pattern, label=label)

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    other_output = open(results_name(some_path, test_number), "a", encoding="utf-8")
    other_output.write(
        merge_type_iteration(some_path, final_model_type, iteration, test_number)
        + " "
        + label
        + ": Threshold = %f, Geometric mean = %.3f" % (thresholds[ix], gmeans[ix])
    )
    other_output.write("\n")
    other_output.close()
    plt.plot(fpr[ix], tpr[ix], "o", markerfacecolor=pattern, markeredgecolor="k")


def make_ROC_plots(
    some_path,
    test_number,
    final_model_type,
    iteration,
    test_labels,
    model_predictions=[],
):
    plt.figure()
    plt.title(
        merge_type_iteration(some_path, final_model_type, iteration, test_number)
        + "\nReceiver operating characteristic (ROC) curve"
    )

    make_a_ROC_plot(
        some_path,
        test_number,
        final_model_type,
        iteration,
        test_labels,
        model_predictions,
        "r",
        "ROC curve",
    )

    # Plot random guessing ROC curve.
    plt.plot([0, 1], [0, 1], "c", label="ROC curve for random guessing")

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig(
        ROC_name(some_path, test_number, final_model_type, iteration),
        bbox_inches="tight",
    )

    plt.close()


# Count correct predictions based on a custom threshold of probability
def my_accuracy_calculate(test_labels, model_predictions, threshold=0.5):
    score = 0

    model_predictions = convert_to_binary(model_predictions, threshold)

    for i in range(len(test_labels)):

        if model_predictions[i] == test_labels[i]:
            score += 1

    return score / len(test_labels) * 100


def output_metrics(
    some_path,
    test_number,
    final_model_type,
    iteration,
    test_labels,
    model_predictions,
    threshold=0.5,
):
    # Get recall and precision.
    precision, recall, _ = precision_recall_curve(test_labels, model_predictions)

    # Convert probabilities to predictions
    model_predictions_binary = convert_to_binary(model_predictions, threshold)

    other_output = open(results_name(some_path, test_number), "a", encoding="utf-8")
    other_output.write(
        "%s: Accuracy: %.2f%% Area under ROC curve: %.4f Area under PR curve: %.4f F1: %.4f"
        % (
            merge_type_iteration(some_path, final_model_type, iteration, test_number),
            my_accuracy_calculate(test_labels, model_predictions, threshold),
            roc_auc_score(test_labels, model_predictions),
            auc(recall, precision),
            f1_score(test_labels, model_predictions_binary),
        )
    )
    other_output.write("\n")
    other_output.close()


def hist_predicted(
    some_path,
    test_number,
    final_model_type,
    iteration,
    test_labels,
    model_predictions,
):
    # Create a histogram of the predicted probabilities only for the peptides that show self-assembly
    SA_name, NSA_name = hist_names(some_path, test_number, final_model_type, iteration)
    model_predictions_true = []
    for x in range(len(test_labels)):
        if test_labels[x] == 1.0:
            model_predictions_true.append(float(model_predictions[x]))

    plt.figure()
    plt.title(
        merge_type_iteration(some_path, final_model_type, iteration, test_number)
        + "\nHistogram of predicted self assembly probability\nfor peptides with self assembly"
    )
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides")
    plt.hist(
        model_predictions_true, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    )

    plt.savefig(SA_name, bbox_inches="tight")
    plt.close()

    # Create a histogram of the predicted probabilities only for the peptides that don't show self-assembly

    model_predictions_false = []
    for x in range(len(test_labels)):
        if test_labels[x] == 0.0:
            model_predictions_false.append(float(model_predictions[x]))

    plt.figure()
    plt.title(
        merge_type_iteration(some_path, final_model_type, iteration, test_number)
        + "\nHistogram of predicted self assembly probability\nfor peptides without self assembly"
    )
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides")
    plt.hist(
        model_predictions_false,
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    )

    plt.savefig(NSA_name, bbox_inches="tight")
    plt.close()

    other_output = open(
        predictions_name(some_path, test_number, final_model_type, iteration),
        "a",
        encoding="utf-8",
    )
    other_output.write(str(model_predictions))
    other_output.write("\n")
    other_output.write(str(test_labels))
    other_output.write("\n")
    other_output.close()


def decorate_stats(some_path, test_number, history, params_nr="", fold_nr=""):
    accuracy = history.history["accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    val_acc = history.history["val_accuracy"]

    accuracy_max = np.max(accuracy)
    val_acc_max = np.max(val_acc)
    loss_min = np.min(loss)
    val_loss_min = np.min(val_loss)

    other_output = open(results_name(some_path, test_number), "a", encoding="utf-8")
    other_output.write(
        "%s: Maximum accuracy = %.2f%% Maximum validation accuracy = %.2f%% Minimal loss = %.2f%% Minimal validation loss = %.2f%%"
        % (
            merge_type_params(some_path, fold_nr, params_nr, test_number),
            accuracy_max * 100,
            val_acc_max * 100,
            loss_min * 100,
            val_loss_min * 100,
        )
    )
    other_output.write("\n")
    other_output.write(
        "%s: Accuracy = %.2f%% (%.2f%%) Validation accuracy = %.2f%% (%.2f%%) Loss = %.2f%% (%.2f%%) Validation loss = %.2f%% (%.2f%%)"
        % (
            merge_type_params(some_path, fold_nr, params_nr, test_number),
            np.mean(accuracy) * 100,
            np.std(accuracy) * 100,
            np.mean(val_acc) * 100,
            np.std(val_acc) * 100,
            np.mean(loss) * 100,
            np.std(loss) * 100,
            np.mean(val_loss) * 100,
            np.std(val_loss) * 100,
        )
    )
    other_output.write("\n")
    other_output.close()

    acc_name, val_acc_name, loss_name, val_loss_nam = history_name(
        some_path, test_number, params_nr, fold_nr
    )

    other_output = open(acc_name, "w", encoding="utf-8")
    other_output.write(str(accuracy))
    other_output.write("\n")
    other_output.close()

    other_output = open(val_acc_name, "w", encoding="utf-8")
    other_output.write(str(val_acc))
    other_output.write("\n")
    other_output.close()

    other_output = open(loss_name, "w", encoding="utf-8")
    other_output.write(str(loss))
    other_output.write("\n")
    other_output.close()

    other_output = open(val_loss_nam, "w", encoding="utf-8")
    other_output.write(str(val_loss))
    other_output.write("\n")
    other_output.close()


def decorate_stats_final(some_path, test_number, history, iteration):
    accuracy = history.history["accuracy"]
    loss = history.history["loss"]

    accuracy_max = np.max(accuracy)
    loss_min = np.min(loss)

    other_output = open(results_name(some_path, test_number), "a", encoding="utf-8")
    other_output.write(
        "%s: Maximum accuracy = %.2f%% Minimal loss = %.2f%%"
        % (
            merge_type_iteration(some_path, "weak", iteration, test_number),
            accuracy_max * 100,
            loss_min * 100,
        )
    )
    other_output.write("\n")
    other_output.write(
        "%s: Accuracy = %.2f%% (%.2f%%) Loss = %.2f%% (%.2f%%)"
        % (
            merge_type_iteration(some_path, "weak", iteration, test_number),
            np.mean(accuracy) * 100,
            np.std(accuracy) * 100,
            np.mean(loss) * 100,
            np.std(loss) * 100,
        )
    )
    other_output.write("\n")
    other_output.close()

    accuracy_name, loss_name = final_history_name(some_path, test_number, iteration)

    other_output = open(accuracy_name, "w", encoding="utf-8")
    other_output.write(str(accuracy))
    other_output.write("\n")
    other_output.close()

    other_output = open(loss_name, "w", encoding="utf-8")
    other_output.write(str(loss))
    other_output.write("\n")
    other_output.close()


def decorate_stats_avg(
    some_path, test_number, accuracy, val_acc, loss, val_loss, params_nr=""
):

    accuracy_max = np.max(accuracy)
    val_acc_max = np.max(val_acc)
    loss_min = np.min(loss)
    val_loss_min = np.min(val_loss)

    other_output = open(results_name(some_path, test_number), "a", encoding="utf-8")
    other_output.write(
        "%s Params %d average: Maximum accuracy = %.2f%% (%.2f%%) Maximum validation accuracy = %.2f%% (%.2f%%) Minimal loss = %.2f%% (%.2f%%) Minimal validation loss = %.2f%% (%.2f%%)"
        % (
            merge_type_test_number(some_path, test_number),
            params_nr,
            np.mean(accuracy_max) * 100,
            np.std(accuracy_max) * 100,
            np.mean(val_acc_max) * 100,
            np.std(val_acc_max) * 100,
            np.mean(loss_min),
            np.std(loss_min) * 100,
            np.mean(val_loss_min),
            np.std(val_loss_min) * 100,
        )
    )
    other_output.write("\n")
    other_output.write(
        "%s Params %d average: Accuracy = %.2f%% (%.2f%%) Validation accuracy = %.2f%% (%.2f%%) Loss = %.2f%% (%.2f%%) Validation loss = %.2f%% (%.2f%%)"
        % (
            merge_type_test_number(some_path, test_number),
            params_nr,
            np.mean(accuracy) * 100,
            np.std(accuracy) * 100,
            np.mean(val_acc) * 100,
            np.std(val_acc) * 100,
            np.mean(loss),
            np.std(loss) * 100,
            np.mean(val_loss),
            np.std(val_loss) * 100,
        )
    )
    other_output.write("\n")
    other_output.close()
