import tensorflow as tf
import numpy as np
import pandas as pd
from custom_plots import plt_model, plt_model_final, decorate_stats, decorate_stats_avg, decorate_stats_final
from utils import convert_list, DATA_PATH, MODEL_DATA_PATH, SEQ_MODEL_DATA_PATH, MY_MODEL_DATA_PATH, scale, split_amino_acids, split_dipeptides, split_tripeptides, padding
import new_model
from sklearn.preprocessing import MinMaxScaler 
from seqprops import SequentialPropertiesEncoder 
import random
from custom_plots import convert_to_binary, make_ROC_plots, make_PR_plots, output_metrics, hist_predicted

LEARNING_RATE_SET = 0.01
MAX_ITERATIONS = 5
MAXLEN = 24

# This function keeps the initial learning rate for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

def return_callbacks(model_file, metric):
    callbacks = [
        # When validation loss doesn't decrease in 10 consecutive epochs, reduce the learning rate by 90%.
        # This is repeated while learning rate is >= 0.000001.
        #tf.keras.callbacks.ReduceLROnPlateau(
        #    monitor='loss', factor=0.1, patience=10, min_lr=0.000001
        #),
        # Save the best model (the one with the lowest validation loss).
        tf.keras.callbacks.ModelCheckpoint(
            model_file, save_best_only=True, monitor=metric, mode='min'
        ),
        # This callback will stop the training when there is no improvement in
        # the loss for three consecutive epochs.
        # Restoring best weights in case of performance drop
        #tf.keras.callbacks.EarlyStopping(monitor=metric, patience=10, restore_best_weights=True),
        tf.keras.callbacks.LearningRateScheduler(scheduler)
    ]
    return callbacks

def read_mordred(sequence_of_peptide, new_props, length_to_pad, masking_value=2):

    df = pd.read_csv(DATA_PATH+"mordred_descriptors.csv", skipinitialspace=True, sep=';')

    index = 0
    
    for i in range(len(df['sequence'])):
        if df['sequence'][i] == sequence_of_peptide:
            index = i
            break
    
    new_props = []
    for column in df.columns[:35]:
        if column == 'sequence':
            continue
        padded = padding([df[column][index]], length_to_pad, masking_value)   
        new_props.append(padded)

    return new_props

def common_multiple_model_predict(thr, model_predictions, alpha_values):
 
    model_predictions_new = []
    
    # Get model predictions on the test data.
    for i in range(len(model_predictions)):
        tmp_model_predictions = []
        for j in range(len(model_predictions[i])):
            tmp_model_predictions.append(model_predictions[i][j])

        for j in range(len(tmp_model_predictions)):
            if tmp_model_predictions[j] > thr:
                tmp_model_predictions[j] = 1
            else:
                tmp_model_predictions[j] = -1
            tmp_model_predictions[j] = tmp_model_predictions[j] * alpha_values[i]
        
        if len(model_predictions_new) == 0:
            for j in range(len(tmp_model_predictions)):
                model_predictions_new.append(tmp_model_predictions[j])
        else:
            for j in range(len(tmp_model_predictions)):
                model_predictions_new[j] += tmp_model_predictions[j]

    for i in range(len(model_predictions_new)):
        if model_predictions_new[i] > 0:
            model_predictions_new[i] = 1
        else:
            model_predictions_new[i] = 0 

    return model_predictions_new 
    
def multiple_model_predict_seq(thr, best_batch_size, test_data, test_labels, models, model_predictions, alpha_values):
    test_data, test_labels = reshape_seq(test_data, test_labels)

    return common_multiple_model_predict(thr, model_predictions, alpha_values)

def multiple_model_predict(thr, num_props, best_batch_size, test_data, test_labels, models, model_predictions, alpha_values):
    test_data, test_labels = reshape(num_props, test_data, test_labels) 

    return common_multiple_model_predict(thr, model_predictions, alpha_values)

def multiple_model_predict_AP(thr, num_props, best_batch_size, test_data, test_labels, models, model_predictions, alpha_values):
    test_data, test_labels = reshape_AP(num_props, test_data, test_labels) 
    
    return common_multiple_model_predict(thr, model_predictions, alpha_values)

def model_predict_seq(best_batch_size, test_data, test_labels, best_model_file, best_model):
    # Load the best model.
    if best_model_file != '':
        best_model = tf.keras.models.load_model(best_model_file) 
    test_data, test_labels = reshape_seq(test_data, test_labels)
    
    # Get model predictions on the test data.
    model_predictions = best_model.predict(test_data, batch_size=best_batch_size)
    model_predictions = convert_list(model_predictions)  
    return model_predictions

def model_predict(num_props, best_batch_size, test_data, test_labels, best_model_file, best_model):
    # Load the best model.
    if best_model_file != '':
        best_model = tf.keras.models.load_model(best_model_file) 
    test_data, test_labels = reshape(num_props, test_data, test_labels) 

    # Get model predictions on the test data.
    model_predictions = best_model.predict(test_data, batch_size=best_batch_size)
    model_predictions = convert_list(model_predictions)  
    return model_predictions

def model_predict_AP(num_props, best_batch_size, test_data, test_labels, best_model_file, best_model):
    # Load the best model.
    if best_model_file != '':
        best_model = tf.keras.models.load_model(best_model_file)
    test_data, test_labels = reshape_AP(num_props, test_data, test_labels) 

    # Get model predictions on the test data.
    model_predictions = best_model.predict(test_data, batch_size=best_batch_size)
    model_predictions = convert_list(model_predictions)  
    return model_predictions

def common_no_file_after_training_first():
    # Get sequences for peptides with no labels and predictions from the model without machine learning 
    resulteval = DATA_PATH+'RESULTEVAL.csv' 
    df = pd.read_csv(resulteval, skipinitialspace=True, sep=';')
    sequences = list(df['Dizajnirani peptid'])
    seq_example = ''
    for i in range(24):
        seq_example += 'A'
    sequences.append(seq_example)
    past_grades = list(df['Postotak svojstava koja imaju AUC > 0,5 koja su dala SA'])
    past_classes= list(df['SA']) 

    SA_data = {}
    for i in range(len(sequences)):
        SA_data[sequences[i]] = '0'

    return SA_data, sequences, past_classes, past_grades

def common_no_file_after_training(thr, sequences, past_classes, past_grades, percentage_filename, grade_filename, model_predictions):
    # Write SA probability to file
    percentage_file = open(percentage_filename, "w", encoding="utf-8")
    percentage_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
     
    for x in range(len(sequences)):
        percentage_string_to_write += sequences[x]+";"+str(np.round(model_predictions[x] * 100, 2))+";"+past_grades[x]+"\n"
    percentage_string_to_write = percentage_string_to_write.replace('.',',')
    percentage_file.write(percentage_string_to_write)
    percentage_file.close()

    # Write class based on the threshold of probability to file
    threshold_amino = thr
    
    model_predictions = convert_to_binary(model_predictions, threshold_amino) 
    
    grade_file = open(grade_filename, "w", encoding="utf-8")
    grade_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    correct_amino = 0 
    for x in range(len(sequences)): 
        if (model_predictions[x] == 1 and past_classes[x] == 'Y') or (model_predictions[x] == 0 and past_classes[x] == 'N'):
            correct_amino += 1 
            
        part1 = sequences[x]+";"+str(model_predictions[x])+";"+past_classes[x]+"\n"
        part1 = part1.replace(".0",'')
        part1 = part1.replace('1','Y')
        part1 = part1.replace('0','N') 
        grade_string_to_write += part1  
    last_line = "Number of matches with method without RNN;"+str(correct_amino)+";\n"
    last_line = last_line.replace(".",',') 
    grade_string_to_write += last_line
    grade_file.write(grade_string_to_write)
    grade_file.close() 

def no_file_after_training_seq(thr, test_number, iteration, model_predictions, properties, names=['AP'], offset = 1, masking_value=2):
    SA_data, sequences, past_classes, past_grades = common_no_file_after_training_first()

    SA, NSA = load_data_SA_seq(SA_data, names, offset, properties, masking_value)
    all_data, all_labels = merge_data_seq(SA, NSA)
    all_data = all_data[:-1] 
    all_labels = all_labels[:-1]  
    sequences = sequences[:-1] 
    
    percentage_filename = SEQ_MODEL_DATA_PATH+str(test_number)+"_"+"percentage_multiple_properties" + iteration + ".csv"
    grade_filename = SEQ_MODEL_DATA_PATH+str(test_number)+"_"+"grade_multiple_properties" + iteration + ".csv"
    
    common_no_file_after_training(thr, sequences, past_classes, past_grades, percentage_filename, grade_filename, model_predictions)

def no_file_after_training(thr, test_number, iteration, model_predictions, properties, names=['AP'], offset = 1, masking_value=2):
    SA_data, sequences, past_classes, past_grades = common_no_file_after_training_first()

    SA, NSA = load_data_SA(SA_data, names, offset, properties, masking_value)
    all_data, all_labels = merge_data(SA, NSA)
    all_data = all_data[:-1] 
    all_labels = all_labels[:-1]  
    sequences = sequences[:-1] 
    
    percentage_filename = MODEL_DATA_PATH+str(test_number)+"_"+"percentage_multiple_properties" + iteration + ".csv"
    grade_filename = MODEL_DATA_PATH+str(test_number)+"_"+"grade_multiple_properties" + iteration + ".csv"
   
    common_no_file_after_training(thr, sequences, past_classes, past_grades, percentage_filename, grade_filename, model_predictions)

def no_file_after_training_AP(thr, test_number, iteration, model_predictions, properties, names=['AP'], offset = 1, masking_value=2):
    SA_data, sequences, past_classes, past_grades = common_no_file_after_training_first()

    SA, NSA = load_data_SA_AP(SA_data, names, offset, properties, masking_value)
    all_data, all_labels = merge_data_AP(SA, NSA)
    all_data = all_data[:-1] 
    all_labels = all_labels[:-1]  
    sequences = sequences[:-1]
    
    percentage_filename = MY_MODEL_DATA_PATH+str(test_number)+"_"+"percentage_multiple_properties" + iteration + ".csv"
    grade_filename = MY_MODEL_DATA_PATH+str(test_number)+"_"+"grade_multiple_properties" + iteration + ".csv"
  
    common_no_file_after_training(thr, sequences, past_classes, past_grades, percentage_filename, grade_filename, model_predictions)

def common_predict(path, model_predictions, test_number, iteration, test_labels): 
    #Plot ROC curves for all models
    make_ROC_plots(path, test_number, iteration, test_labels, model_predictions)
    
    #Plot PR curves for all models
    make_PR_plots(path, test_number, iteration, test_labels, model_predictions)

    # Output adjusted accuracy, F1 score and ROC AUC score for all models
    output_metrics(test_labels, model_predictions) 
    
    # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
    hist_predicted(path, test_number, iteration, test_labels, model_predictions) 

def adaboost_generate_predictions(thr, model_predictions, test_number, iteration, test_labels, properties, names = ['AP'], offset = 1, masking_value = 2):
    # Generate predictions on data that has no label beforehand 
    common_predict(MODEL_DATA_PATH, model_predictions, test_number, iteration, test_labels)
    no_file_after_training(thr, test_number, iteration, model_predictions, properties, names, offset, masking_value)
 
def adaboost_generate_predictions_seq(thr, model_predictions, test_number, iteration, test_labels, properties, names = ['AP'], offset = 1, masking_value = 2):
    # Generate predictions on data that has no label beforehand 
    common_predict(SEQ_MODEL_DATA_PATH, model_predictions, test_number, iteration, test_labels)
    no_file_after_training_seq(thr, test_number, iteration, model_predictions, properties, names, offset, masking_value)
 
def adaboost_generate_predictions_AP(thr, model_predictions, test_number, iteration, test_labels, properties, names = ['AP'], offset = 1, masking_value = 2): 
    # Generate predictions on data that has no label beforehand 
    common_predict(MY_MODEL_DATA_PATH, model_predictions, test_number, iteration, test_labels)
    no_file_after_training_AP(thr, test_number, iteration, model_predictions, properties, names, offset, masking_value)
 
def extract_len_from_data_and_labels(data, labels, len_target, padding):
    only_one_len_indices = [] 
    other_len_indices = [] 
    for index in range(len(data)):
        index_start_padding = 0
        for padding_begin in range(len(data[index][0])):
            if data[index][0][padding_begin] == padding:
                index_start_padding = padding_begin
                break
        if index_start_padding == len_target:
            only_one_len_indices.append(index) 
        else: 
            other_len_indices.append(index) 
            
    return only_one_len_indices, other_len_indices
            
def data_and_labels_from_indices(all_data, all_labels, indices):
    data = []
    labels = []

    for i in indices:
        data.append(all_data[i])
        labels.append(all_labels[i]) 
        
    return data, labels 

def reshape_seq(all_data, all_labels):
    data = []
    labels = []

    for i in range(len(all_data)):
        data.append(all_data[i])
        labels.append(all_labels[i]) 
    if len(data) > 0:
        data = np.reshape(data, (len(data), np.shape(data[0])[0], np.shape(data[0])[1]))
    labels = np.array(labels)
    
    return data, labels 

def reshape_AP(num_props, all_data, all_labels):
    data = [[] for i in range(len(all_data[0]))]
    labels = []
    for i in range(len(all_data)):
        for j in range(len(all_data[0])):
            data[j].append(all_data[i][j])
        labels.append(all_labels[i])  
    new_data = []   
    for i in range(len(data)):
        if len(data[i]) > 0 and i < num_props:
            new_data.append(np.array(data[i]).reshape(len(labels), -1))  
    labels = np.array(labels) 
    return new_data, labels 

def reshape(num_props, all_data, all_labels):
    data = [[] for i in range(len(all_data[0]))]
    labels = []
    for i in range(len(all_data)):
        for j in range(len(all_data[0])):
            data[j].append(all_data[i][j])
        labels.append(all_labels[i])  
    new_data = []
    last_data = []    
    for i in range(len(data)):
        if len(data[i]) > 0 and i < num_props:
            new_data.append(np.array(data[i]).reshape(len(labels), -1))
        if len(data[i]) > 0 and i >= num_props:
           last_data.append(np.array(data[i]).reshape(len(labels), -1))
    last_data = np.array(last_data).reshape(len(last_data[0]), len(last_data[0][0]), len(last_data))
    new_data.append(last_data)
    labels = np.array(labels) 
    return new_data, labels 

# Choose loading AP, APH, logP or polarity
def load_data_AP(name = 'AP', offset = 1):
    # Load AP scores. 
    amino_acids_AP = np.load(DATA_PATH+'amino_acids_'+name+'.npy', allow_pickle=True).item()
    dipeptides_AP = np.load(DATA_PATH+'dipeptides_'+name+'.npy', allow_pickle=True).item()
    tripeptides_AP = np.load(DATA_PATH+'tripeptides_'+name+'.npy', allow_pickle=True).item()
    
    # Scale scores to range [-1, 1].
    scale(amino_acids_AP, offset)
    scale(dipeptides_AP, offset)
    scale(tripeptides_AP, offset)

    return amino_acids_AP, dipeptides_AP, tripeptides_AP

def read_from_npy_SA(SA_data):
    sequences = []
    labels = []
    for peptide in SA_data:
        if len(peptide) > MAXLEN:
            continue
        sequences.append(peptide)
        labels.append(SA_data[peptide])

    return sequences, labels

def load_data_SA_seq(SA_data, names=['AP'], offset = 1, properties_to_include = [], masking_value=2):
    sequences, labels = read_from_npy_SA(SA_data)
            
    # Encode sequences
    encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-offset, offset)))
    encoded_sequences = encoder.encode(sequences)
     
    # Split peptides in two bins.
    # SA - has self-assembly, NSA - does not have self-assembly.
    # Convert each peptide sequence to three lists of AP scores: amino acid, dipeptide and tripeptide scores.
    SA = []
    NSA = []
    for index in range(len(sequences)):
        new_props = []
        for name in names:
            amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP(name, offset)  
            amino_acids_ap = split_amino_acids(sequences[index], amino_acids_AP)
            dipeptides_ap = split_dipeptides(sequences[index], dipeptides_AP)
            tripeptides_ap = split_tripeptides(sequences[index], tripeptides_AP)
                    
            amino_acids_ap_padded = padding(amino_acids_ap, len(encoded_sequences[index]), masking_value)
            dipeptides_acids_ap_padded = padding(dipeptides_ap, len(encoded_sequences[index]), masking_value)
            tripeptides_ap_padded = padding(tripeptides_ap, len(encoded_sequences[index]), masking_value)  

            new_props.append(amino_acids_ap_padded)
            new_props.append(dipeptides_acids_ap_padded)
            new_props.append(tripeptides_ap_padded) 

            #new_props = read_mordred(sequences[index], new_props, len(encoded_sequences[index]), masking_value)
        
        other_props = np.reshape(encoded_sequences[index], (len(encoded_sequences[index][0]), len(encoded_sequences[index])))
                                 
        for prop_index in range(len(properties_to_include)):
            if prop_index < len(other_props) and properties_to_include[prop_index] == 1:
                array = other_props[prop_index]
                for i in range(len(array)):
                    if array[i] == 0.0:
                        array[i] = 2.0
                new_props.append(array)
                 
        new_props = np.reshape(new_props, (len(encoded_sequences[index]), len(new_props))) 
        
        if labels[index] == '1':
            SA.append(new_props) 
        elif labels[index] == '0':
            NSA.append(new_props) 
    if len(SA) > 0:
        SA = np.reshape(SA, (len(SA), np.shape(SA[0])[0], np.shape(SA[0])[1]))
    if len(NSA) > 0:
        NSA = np.reshape(NSA, (len(NSA), np.shape(NSA[0])[0], np.shape(NSA[0])[1]))
    return SA, NSA

def load_data_SA(SA_data, names=['AP'], offset = 1, properties_to_include = [], masking_value=2):
    sequences, labels = read_from_npy_SA(SA_data)
            
    # Encode sequences
    encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-offset, offset)))
    encoded_sequences = encoder.encode(sequences)
     
    # Split peptides in two bins.
    # SA - has self-assembly, NSA - does not have self-assembly.
    # Convert each peptide sequence to three lists of AP scores: amino acid, dipeptide and tripeptide scores.
    SA = []
    NSA = []
    for index in range(len(sequences)):
        new_props = []
        for name in names:
            amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP(name, offset)  
            amino_acids_ap = split_amino_acids(sequences[index], amino_acids_AP)
            dipeptides_ap = split_dipeptides(sequences[index], dipeptides_AP)
            tripeptides_ap = split_tripeptides(sequences[index], tripeptides_AP)
                    
            amino_acids_ap_padded = padding(amino_acids_ap, len(encoded_sequences[index]), masking_value)
            dipeptides_acids_ap_padded = padding(dipeptides_ap, len(encoded_sequences[index]), masking_value)
            tripeptides_ap_padded = padding(tripeptides_ap, len(encoded_sequences[index]), masking_value)  
        
            new_props.append(amino_acids_ap_padded)
            new_props.append(dipeptides_acids_ap_padded)
            new_props.append(tripeptides_ap_padded)

            #new_props = read_mordred(sequences[index], new_props, len(encoded_sequences[index]), masking_value)
        
        other_props = np.reshape(encoded_sequences[index], (len(encoded_sequences[index][0]), len(encoded_sequences[index])))
                                 
        for prop_index in range(len(properties_to_include)):
            if prop_index < len(other_props) and properties_to_include[prop_index] == 1:
                array = other_props[prop_index]
                for i in range(len(array)):
                    if array[i] == 0.0:
                        array[i] = 2.0
                new_props.append(array) 
        
        if labels[index] == '1':
            SA.append(new_props) 
        elif labels[index] == '0':
            NSA.append(new_props) 
            
    return SA, NSA

def load_data_SA_AP(SA_data, names=['AP'], offset = 1, properties_to_include = [], masking_value=2):
    sequences, labels = read_from_npy_SA(SA_data)
     
    # Split peptides in two bins.
    # SA - has self-assembly, NSA - does not have self-assembly.
    # Convert each peptide sequence to three lists of AP scores: amino acid, dipeptide and tripeptide scores.
    SA = []
    NSA = []
    for index in range(len(sequences)):
        new_props = []
        for name in names:
            amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP(name, offset)  
            amino_acids_ap = split_amino_acids(sequences[index], amino_acids_AP)
            dipeptides_ap = split_dipeptides(sequences[index], dipeptides_AP)
            tripeptides_ap = split_tripeptides(sequences[index], tripeptides_AP)
                    
            amino_acids_ap_padded = padding(amino_acids_ap, MAXLEN, masking_value)
            dipeptides_acids_ap_padded = padding(dipeptides_ap, MAXLEN, masking_value)
            tripeptides_ap_padded = padding(tripeptides_ap, MAXLEN, masking_value)  
        
            new_props.append(amino_acids_ap_padded)
            new_props.append(dipeptides_acids_ap_padded)
            new_props.append(tripeptides_ap_padded) 

            #new_props = read_mordred(sequences[index], new_props, MAXLEN, masking_value)
        
        if labels[index] == '1':
            SA.append(new_props) 
        elif labels[index] == '0':
            NSA.append(new_props) 
            
    return SA, NSA

def merge_data(SA, NSA):
    # Merge the bins and add labels
    merged_data = []
    for i in SA:
        merged_data.append(i)
    for i in NSA:
        merged_data.append(i)

    merged_labels = np.ones(len(SA) + len(NSA))
    merged_labels[len(SA):] *= 0
    return merged_data, merged_labels
   
def merge_data_AP(SA, NSA):
    # Merge the bins and add labels
    merged_data = []
    for i in SA:
        merged_data.append(i)
    for i in NSA:
        merged_data.append(i)

    merged_labels = np.ones(len(SA) + len(NSA))
    merged_labels[len(SA):] *= 0
    return merged_data, merged_labels

def merge_data_seq(SA, NSA):
    # Merge the bins and add labels
    merged_data = []
    for i in SA:
        merged_data.append(i)
    for i in NSA:
        merged_data.append(i)
    if len(merged_data) > 0:
        merged_data = np.reshape(merged_data, (len(merged_data), np.shape(merged_data[0])[0], np.shape(merged_data[0])[1]))
    merged_labels = np.ones(len(SA) + len(NSA))
    merged_labels[len(SA):] *= 0

    return merged_data, merged_labels

def model_training_seq(test_number, train_and_validation_data, train_and_validation_labels, kfold_second, epochs, factor_NSA, test_data, test_labels, properties, names, offset, mask_value=2):
    
    model_name = "multiple_properties" 
           
    params_nr = 0 
    min_val_loss = 1000
    
    hyperparameter_conv = [5]
    hyperparameter_numcells = [32, 48, 64] 
    hyperparameter_kernel_size = [4, 6, 8]
    hyperparameter_dropout = [0.5]
    hyperparameter_batch_size = [600]
    
    best_conv = 0
    best_numcells = 0
    best_kernel = 0
    best_batch_size = 0
    best_dropout = 0
    
    indices = []
    for train_data_indices, validation_data_indices in kfold_second.split(train_and_validation_data, train_and_validation_labels): 
        indices.append([train_data_indices, validation_data_indices])
        
    for conv in hyperparameter_conv:
        for numcells in hyperparameter_numcells:
            for kernel in hyperparameter_kernel_size: 
                for batch in hyperparameter_batch_size: 
                    for dropout in hyperparameter_dropout: 
                        params_nr += 1
                        fold_nr = 0 
                        history_val_loss = []
                        history_val_acc = []
                        history_loss = []
                        history_acc = []
                        
                        for pair in indices:   
                            
                            train_data_indices = pair[0]
                            
                            validation_data_indices = pair[1]  
                            
                            fold_nr += 1 
    
                            # Convert train indices to train data and train labels
                            train_data, train_labels = data_and_labels_from_indices(train_and_validation_data, train_and_validation_labels, train_data_indices)
                            
                            train_data, train_labels = reshape_seq(train_data, train_labels)
                            
                            # Convert validation indices to validation data and validation labels
                            val_data, val_labels = data_and_labels_from_indices(train_and_validation_data, train_and_validation_labels, validation_data_indices)
                            
                            val_data, val_labels = reshape_seq(val_data, val_labels)                    
                               
                            # Save model to correct file based on number of fold
                            model_picture = SEQ_MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_'+str(params_nr)+'_'+str(fold_nr)+'.png'
                            model_file = SEQ_MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_'+str(params_nr)+'_'+str(fold_nr)+'.h5'
                            
                            #  Choose correct model and instantiate model
                            model = new_model.create_seq_model(input_shape=np.shape(train_data[0]), conv1_filters=conv, conv2_filters=conv, conv_kernel_size=kernel, num_cells=numcells, dropout=dropout, mask_value=mask_value)
                    
                            # Save graphical representation of the model to a file.
                            tf.keras.utils.plot_model(model, to_file=model_picture, show_shapes=True)
                            
                            # Print model summary.
                            model.summary()
                            
                            callbacks = return_callbacks(model_file, 'val_loss') 
                    
                            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_SET)
                    
                            model.compile(
                                optimizer=optimizer,
                                loss='binary_crossentropy',
                                metrics=['accuracy']
                            ) 
                            
                            # Train the model.
                            # After model training, the `history` variable will contain important parameters for each epoch, such as
                            # train loss, train accuracy, learning rate, and so on.
                            history = model.fit(
                                train_data,
                                train_labels,
                                #validation_split = 0.1,
                                validation_data=[val_data, val_labels],
                                class_weight={0: factor_NSA, 1: 1.0},
                                epochs=epochs,
                                batch_size = batch,
                                callbacks=callbacks,
                                verbose=1
                            )
                            
                            history_val_loss += history.history['val_loss']
                            history_val_acc += history.history['val_accuracy']
                            history_loss += history.history['loss']
                            history_acc += history.history['accuracy']
                            
                            # Output accuracy, validation accuracy, loss and validation loss for all models
                            decorate_stats(history, params_nr, fold_nr)
                    
                            # Plot the history
                            plt_model(SEQ_MODEL_DATA_PATH, test_number, history, 'rnn_model_'+model_name+'_'+str(params_nr)+'_'+str(fold_nr))
                            
                        # Output accuracy, validation accuracy, loss and validation loss for all models
                        print('Test %d testing params %d on fold %d: conv: %d num_cells: %d kernel_size: %d batch_size: %d dropout: %.2f' % (test_number, params_nr, fold_nr, conv, numcells, kernel, batch, dropout))
                        decorate_stats_avg(history_acc, history_val_acc, history_loss, history_val_loss, params_nr)
                        avg_val_loss = np.mean(history_val_loss)  
                        
                        if avg_val_loss < min_val_loss:
                            min_val_loss = avg_val_loss
                            best_conv = conv
                            best_numcells = numcells
                            best_kernel = kernel
                            best_batch_size = batch
                            best_dropout = dropout 
                    
    # All samples have equal weights 
    sample_weights = []
    for i in range(len(train_and_validation_labels)):
        sample_weights.append(1 / len(train_and_validation_labels))  
            
    final_train_seq([], [], [], sample_weights, 1, factor_NSA, epochs, test_number, model_name, train_and_validation_data, train_and_validation_labels, best_batch_size, best_dropout, best_conv, best_numcells, best_kernel, test_data, test_labels, properties, names, offset, mask_value)

def model_training(num_props, test_number, train_and_validation_data, train_and_validation_labels, kfold_second, epochs, factor_NSA, test_data, test_labels, properties, names, offset, mask_value=2):
    
    model_name = "multiple_properties" 
           
    params_nr = 0 
    min_val_loss = 1000
    
    hyperparameter_conv = [5]
    hyperparameter_numcells = [32, 48, 64]
    hyperparameter_kernel_size = [4, 6, 8]
    hyperparameter_lstm = [5]
    hyperparameter_dense = [15]
    hyperparameter_lambda = [0.0]
    hyperparameter_dropout = [0.5]
    hyperparameter_batch_size = [600]
    
    best_conv = 0
    best_numcells = 0
    best_kernel = 0
    best_lstm = 0
    best_dense = 0
    best_lambda = 0
    best_dropout = 0
    best_batch_size = 0
    
    indices = []
    for train_data_indices, validation_data_indices in kfold_second.split(train_and_validation_data, train_and_validation_labels): 
        indices.append([train_data_indices, validation_data_indices])
   
    for conv in hyperparameter_conv:
        for numcells in hyperparameter_numcells: 
            for kernel in hyperparameter_kernel_size: 
                for lstm_NEW in hyperparameter_lstm:
                    for dense_NEW in hyperparameter_dense:
                        for my_lambda in hyperparameter_lambda: 
                            for dropout in hyperparameter_dropout: 
                                for batch in hyperparameter_batch_size: 
                                    params_nr += 1
                                    fold_nr = 0 
                                    history_val_loss = []
                                    history_val_acc = []
                                    history_loss = []
                                    history_acc = []
                                    
                                    lstm = conv
                                    dense = numcells * 2
                                    
                                    for pair in indices:   
                                        
                                        train_data_indices = pair[0]
                                        
                                        validation_data_indices = pair[1]
                                        
                                        fold_nr += 1 
                
                                        # Convert train indices to train data and train labels
                                        train_data, train_labels = data_and_labels_from_indices(train_and_validation_data, train_and_validation_labels, train_data_indices)
                                        
                                        train_data, train_labels = reshape(num_props, train_data, train_labels)
                                        
                                        # Convert validation indices to validation data and validation labels
                                        val_data, val_labels = data_and_labels_from_indices(train_and_validation_data, train_and_validation_labels, validation_data_indices)
                                        
                                        val_data, val_labels = reshape(num_props, val_data, val_labels)
                                        
                                        # Save model to correct file based on number of fold
                                        model_picture = MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_'+str(params_nr)+'_'+str(fold_nr)+'.png'
                                        model_file = MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_'+str(params_nr)+'_'+str(fold_nr)+'.h5'
                                        
                                        #  Choose correct model and instantiate model 
                                        model = new_model.amino_di_tri_model(num_props, input_shape=np.shape(train_data[num_props][0]), conv=conv, numcells=numcells, kernel_size=kernel, lstm1=lstm, lstm2=lstm, dense=dense, dropout=dropout, lambda2=my_lambda, mask_value=mask_value)
                  
                                        # Save graphical representation of the model to a file.
                                        tf.keras.utils.plot_model(model, to_file=model_picture, show_shapes=True)
                                        
                                        # Print model summary.
                                        model.summary() 

                                        callbacks = return_callbacks(model_file, 'val_loss') 
                                
                                        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_SET)
                                
                                        model.compile(
                                            optimizer=optimizer,
                                            loss='binary_crossentropy',
                                            metrics=['accuracy']
                                        ) 
                                        
                                        # Train the model.
                                        # After model training, the `history` variable will contain important parameters for each epoch, such as
                                        # train loss, train accuracy, learning rate, and so on.
                                        history = model.fit(
                                            train_data,
                                            train_labels,
                                            #validation_split = 0.1,
                                            validation_data=[val_data, val_labels],
                                            class_weight={0: factor_NSA, 1: 1.0},
                                            epochs=epochs,
                                            batch_size = batch,
                                            callbacks=callbacks,
                                            verbose=1
                                        )
                                        
                                        history_val_loss += history.history['val_loss']
                                        history_val_acc += history.history['val_accuracy']
                                        history_loss += history.history['loss']
                                        history_acc += history.history['accuracy']
                                        
                                        # Output accuracy, validation accuracy, loss and validation loss for all models
                                        decorate_stats(history, params_nr, fold_nr)
                                
                                        # Plot the history
                                        plt_model(MODEL_DATA_PATH, test_number, history, 'rnn_model_'+model_name+'_'+str(params_nr)+'_'+str(fold_nr))
                                    
                                    # Output accuracy, validation accuracy, loss and validation loss for all models
                                    print("Test %d testing params %d on fold %d: conv: %d num_cells: %d kernel_size: %d lstm: %d dense: %d lambda: %.2f dropout: %.2f batch: %d" % (test_number, params_nr, fold_nr, conv, numcells, kernel, lstm, dense, my_lambda, dropout, batch))
                                    decorate_stats_avg(history_acc, history_val_acc, history_loss, history_val_loss, params_nr)
                                    avg_val_loss = np.mean(history_val_loss)  
                                    
                                    if avg_val_loss < min_val_loss:
                                        min_val_loss = avg_val_loss
                                        best_conv = conv
                                        best_numcells = numcells
                                        best_kernel = kernel
                                        best_lstm = lstm
                                        best_dense = dense
                                        best_lambda = my_lambda
                                        best_dropout = dropout
                                        best_batch_size = batch
                                   
    # All samples have equal weights 
    sample_weights = []
    for i in range(len(train_and_validation_labels)):
        sample_weights.append(1 / len(train_and_validation_labels))  
            
    final_train([], [], [], sample_weights, 1, factor_NSA, epochs, test_number, model_name, num_props, train_and_validation_data, train_and_validation_labels, best_batch_size, best_lstm, best_dense, best_dropout, best_lambda, best_conv, best_numcells, best_kernel, test_data, test_labels, properties, names, offset, mask_value)

def model_training_AP(num_props, test_number, train_and_validation_data, train_and_validation_labels, kfold_second, epochs, factor_NSA, test_data, test_labels, properties, names, offset, mask_value=2):

    model_name = "multiple_properties" 
           
    params_nr = 0 
    min_val_loss = 1000
      
    hyperparameter_lstm = [5]
    hyperparameter_dense = [64, 96, 128]
    hyperparameter_lambda = [0.0]
    hyperparameter_dropout = [0.5]
    hyperparameter_batch_size = [600]
     
    best_lstm = 0
    best_dense = 0
    best_lambda = 0
    best_dropout = 0
    best_batch_size = 0
    
    indices = []
    for train_data_indices, validation_data_indices in kfold_second.split(train_and_validation_data, train_and_validation_labels): 
        indices.append([train_data_indices, validation_data_indices])
   
    for lstm in hyperparameter_lstm:
        for dense in hyperparameter_dense:
            for my_lambda in hyperparameter_lambda: 
                for dropout in hyperparameter_dropout: 
                    for batch in hyperparameter_batch_size: 
                        params_nr += 1
                        fold_nr = 0 
                        history_val_loss = []
                        history_val_acc = []
                        history_loss = []
                        history_acc = []
                        
                        for pair in indices:   
                            
                            train_data_indices = pair[0]
                            
                            validation_data_indices = pair[1]
                            
                            fold_nr += 1 
    
                            # Convert train indices to train data and train labels
                            train_data, train_labels = data_and_labels_from_indices(train_and_validation_data, train_and_validation_labels, train_data_indices)
                            
                            train_data, train_labels = reshape_AP(num_props, train_data, train_labels)
                            
                            # Convert validation indices to validation data and validation labels
                            val_data, val_labels = data_and_labels_from_indices(train_and_validation_data, train_and_validation_labels, validation_data_indices)
                            
                            val_data, val_labels = reshape_AP(num_props, val_data, val_labels)
                            
                            # Save model to correct file based on number of fold
                            model_picture = MY_MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_'+str(params_nr)+'_'+str(fold_nr)+'.png'
                            model_file = MY_MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_'+str(params_nr)+'_'+str(fold_nr)+'.h5'
                            
                            #  Choose correct model and instantiate model 
                            model = new_model.only_amino_di_tri_model(num_props, lstm1=lstm, lstm2=lstm, dense=dense, dropout=dropout, lambda2=my_lambda, mask_value=mask_value)
                                
                            # Save graphical representation of the model to a file.
                            tf.keras.utils.plot_model(model, to_file=model_picture, show_shapes=True)
                            
                            # Print model summary.
                            model.summary()
                            
                            callbacks = return_callbacks(model_file, 'val_loss') 
                    
                            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_SET)
                    
                            model.compile(
                                optimizer=optimizer,
                                loss='binary_crossentropy',
                                metrics=['accuracy']
                            ) 
                            
                            # Train the model.
                            # After model training, the `history` variable will contain important parameters for each epoch, such as
                            # train loss, train accuracy, learning rate, and so on.
                            history = model.fit(
                                train_data,
                                train_labels,
                                #validation_split = 0.1,
                                validation_data=[val_data, val_labels],
                                class_weight={0: factor_NSA, 1: 1.0},
                                epochs=epochs,
                                batch_size = batch,
                                callbacks=callbacks,
                                verbose=1
                            )
                            
                            history_val_loss += history.history['val_loss']
                            history_val_acc += history.history['val_accuracy']
                            history_loss += history.history['loss']
                            history_acc += history.history['accuracy']
                            
                            # Output accuracy, validation accuracy, loss and validation loss for all models
                            decorate_stats(history, params_nr, fold_nr)
                    
                            # Plot the history
                            plt_model(MY_MODEL_DATA_PATH, test_number, history, 'rnn_model_'+model_name+'_'+str(params_nr)+'_'+str(fold_nr))
                         
                        # Output accuracy, validation accuracy, loss and validation loss for all models
                        print("Test %d testing params %d on fold %d: lstm: %d dense: %d lambda: %.2f dropout: %.2f batch: %d" % (test_number, params_nr, fold_nr, lstm, dense, my_lambda, dropout, batch))
                        decorate_stats_avg(history_acc, history_val_acc, history_loss, history_val_loss, params_nr)
                        avg_val_loss = np.mean(history_val_loss)  
                        
                        if avg_val_loss < min_val_loss:
                            min_val_loss = avg_val_loss 
                            best_lstm = lstm
                            best_dense = dense
                            best_lambda = my_lambda
                            best_dropout = dropout
                            best_batch_size = batch
                        
    # All samples have equal weights 
    sample_weights = []
    for i in range(len(train_and_validation_labels)):
        sample_weights.append(1 / len(train_and_validation_labels))  
            
    final_train_AP([], [], [], sample_weights, 1, factor_NSA, epochs, test_number, model_name, num_props, train_and_validation_data, train_and_validation_labels, best_batch_size, best_lstm, best_dense, best_dropout, best_lambda, test_data, test_labels, properties, names, offset, mask_value)
    
def final_train_AP(models, model_predictions, alpha_values, sample_weights, iteration, factor_NSA, epochs, test_number, model_name, num_props, data, labels, best_batch_size, best_lstm, best_dense, best_dropout, best_lambda, test_data, test_labels, properties, names, offset, mask_value):
                    
    model_picture = MY_MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_final_model_iteration'+str(iteration)+'.png'
    model_file = MY_MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_final_model_iteration'+str(iteration)+'.h5'
    
    train_and_validation_data, train_and_validation_labels = reshape_AP(num_props, data, labels)
    
    #  Choose correct model and instantiate model
    model = new_model.only_amino_di_tri_model(num_props, lstm1=best_lstm, lstm2=best_lstm, dense=best_dense, dropout=best_dropout, lambda2=best_lambda, mask_value=mask_value)
 
    # Save graphical representation of the model to a file.
    tf.keras.utils.plot_model(model, to_file=model_picture, show_shapes=True)
    
    # Print model summary.
    model.summary()
    
    callbacks = return_callbacks(model_file, 'loss') 

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_SET)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    ) 
    
    count_len_SA = 0
    count_len_NSA = 0
    for label in train_and_validation_labels:
        if label == 1:
            count_len_SA += 1
        else:
            count_len_NSA += 1
    factor_NSA = count_len_SA / count_len_NSA 

    # Train the model.
    # After model training, the `history` variable will contain important parameters for each epoch, such as
    # train loss, train accuracy, learning rate, and so on.
    history = model.fit(
        train_and_validation_data, 
        train_and_validation_labels,
        #validation_split = 0.1, 
        class_weight={0: factor_NSA, 1: 1.0},
        sample_weight=np.reshape(sample_weights, (len(sample_weights), )),
        epochs=epochs,
        batch_size = best_batch_size,
        callbacks=callbacks,
        verbose=1
    ) 
                    
    print("Test %d Iteration %d best params: lstm: %d dense: %d lambda: %.2f dropout: %.2f batch: %d" % (test_number, iteration, best_lstm, best_dense, best_lambda, best_dropout, best_batch_size))
    
    # Output accuracy, validation accuracy, loss and validation loss for all models
    decorate_stats_final(history) 

    # Plot the history
    plt_model_final(MY_MODEL_DATA_PATH, test_number, history, 'rnn_model_'+model_name+'_final_model_iteration'+str(iteration))
    
    best_model_file = MY_MODEL_DATA_PATH+str(test_number)+'_rnn_model_multiple_properties_final_model_iteration'+str(iteration)+'.h5' 
    
    alpha1, data, labels, sample_weights = boost_AP(num_props, best_batch_size, data, labels, best_model_file, sample_weights, 0.5)
    
    models.append(model)
    alpha_values.append(alpha1)
    model_pred_single = model_predict_AP(num_props, best_batch_size, test_data, test_labels, '', model) 
    model_predictions.append(model_pred_single)
    model_pred_multiple = multiple_model_predict_AP(0.5, num_props, best_batch_size, test_data, test_labels, models, model_predictions, alpha_values)

    adaboost_generate_predictions_AP(0.5, model_pred_single, test_number, "_weak" + str(iteration), test_labels, properties, names, offset, mask_value)

    adaboost_generate_predictions_AP(0.5, model_pred_multiple, test_number, "_iteration" + str(iteration), test_labels, properties, names, offset, mask_value)
    
    if iteration < MAX_ITERATIONS:
        final_train_AP(models, model_predictions, alpha_values, sample_weights, iteration + 1, factor_NSA, epochs, test_number, model_name, num_props, data, labels, best_batch_size, best_lstm, best_dense, best_dropout, best_lambda, test_data, test_labels, properties, names, offset,  mask_value)
   
def final_train(models, model_predictions, alpha_values, sample_weights, iteration, factor_NSA, epochs, test_number, model_name, num_props, data, labels, best_batch_size, best_lstm, best_dense, best_dropout, best_lambda, best_conv, best_numcells, best_kernel, test_data, test_labels, properties, names, offset, mask_value):
                        
    model_picture = MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_final_model_iteration'+str(iteration)+'.png'
    model_file = MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_final_model_iteration'+str(iteration)+'.h5'
    
    train_and_validation_data, train_and_validation_labels = reshape(num_props, data, labels)
    
    #  Choose correct model and instantiate model
    model = new_model.amino_di_tri_model(num_props, input_shape = np.shape(train_and_validation_data[num_props][0]), conv=best_conv, numcells=best_numcells, kernel_size = best_kernel,  lstm1=best_lstm, lstm2=best_lstm, dense=best_dense, dropout=best_dropout, lambda2=best_lambda, mask_value=mask_value)
 
    # Save graphical representation of the model to a file.
    tf.keras.utils.plot_model(model, to_file=model_picture, show_shapes=True)
    
    # Print model summary.
    model.summary()
    
    callbacks = return_callbacks(model_file, 'loss') 

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_SET)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    ) 
    
    count_len_SA = 0
    count_len_NSA = 0
    for label in train_and_validation_labels:
        if label == 1:
            count_len_SA += 1
        else:
            count_len_NSA += 1
    factor_NSA = count_len_SA / count_len_NSA 

    # Train the model.
    # After model training, the `history` variable will contain important parameters for each epoch, such as
    # train loss, train accuracy, learning rate, and so on.
    history = model.fit(
        train_and_validation_data, 
        train_and_validation_labels,
        #validation_split = 0.1, 
        class_weight={0: factor_NSA, 1: 1.0},
        sample_weight=np.reshape(sample_weights, (len(sample_weights), )),
        epochs=epochs,
        batch_size = best_batch_size,
        callbacks=callbacks,
        verbose=1
    ) 
                    
    print("Test %d Iteration %d best params: conv: %d num_cells: %d kernel_size: %d lstm: %d dense: %d lambda: %.2f dropout: %.2f batch: %d" % (test_number, iteration, best_conv, best_numcells, best_kernel, best_lstm, best_dense, best_lambda, best_dropout, best_batch_size))
    
    # Output accuracy, validation accuracy, loss and validation loss for all models
    decorate_stats_final(history) 

    # Plot the history
    plt_model_final(MODEL_DATA_PATH, test_number, history, 'rnn_model_'+model_name+'_final_model_iteration'+str(iteration))
    
    best_model_file = MODEL_DATA_PATH+str(test_number)+'_rnn_model_multiple_properties_final_model_iteration'+str(iteration)+'.h5' 
    
    alpha1, data, labels, sample_weights = boost(num_props, best_batch_size, data, labels, best_model_file, sample_weights, 0.5)
    
    models.append(model)
    alpha_values.append(alpha1)
    model_pred_single = model_predict(num_props, best_batch_size, test_data, test_labels, '', model) 
    model_predictions.append(model_pred_single)
    model_pred_multiple = multiple_model_predict(0.5, num_props, best_batch_size, test_data, test_labels, models, model_predictions, alpha_values)

    adaboost_generate_predictions(0.5, model_pred_single, test_number, "_weak" + str(iteration), test_labels, properties, names, offset, mask_value)

    adaboost_generate_predictions(0.5, model_pred_multiple, test_number, "_iteration" + str(iteration), test_labels, properties, names, offset, mask_value)

    if iteration < MAX_ITERATIONS:
        final_train(models, model_predictions, alpha_values, sample_weights, iteration + 1, factor_NSA, epochs, test_number, model_name, num_props, data, labels, best_batch_size, best_lstm, best_dense, best_dropout, best_lambda, best_conv, best_numcells, best_kernel, test_data, test_labels, properties, names, offset,  mask_value)

def final_train_seq(models, model_predictions, alpha_values, sample_weights, iteration, factor_NSA, epochs, test_number, model_name, data, labels, best_batch_size, best_dropout, best_conv, best_numcells, best_kernel, test_data, test_labels, properties, names, offset, mask_value):
                        
    model_picture = SEQ_MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_final_model_iteration'+str(iteration)+'.png'
    model_file = SEQ_MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_final_model_iteration'+str(iteration)+'.h5'
    
    train_and_validation_data, train_and_validation_labels = reshape_seq(data, labels)
    
    #  Choose correct model and instantiate model
    model = new_model.create_seq_model(input_shape=np.shape(train_and_validation_data[0]), conv1_filters=best_conv, conv2_filters=best_conv, conv_kernel_size=best_kernel, num_cells=best_numcells, dropout=best_dropout, mask_value=mask_value)
 
    # Save graphical representation of the model to a file.
    tf.keras.utils.plot_model(model, to_file=model_picture, show_shapes=True)
    
    # Print model summary.
    model.summary()
    
    callbacks = return_callbacks(model_file, 'loss') 

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_SET)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    ) 
    
    count_len_SA = 0
    count_len_NSA = 0
    for label in train_and_validation_labels:
        if label == 1:
            count_len_SA += 1
        else:
            count_len_NSA += 1
    factor_NSA = count_len_SA / count_len_NSA 

    # Train the model.
    # After model training, the `history` variable will contain important parameters for each epoch, such as
    # train loss, train accuracy, learning rate, and so on.
    history = model.fit(
        train_and_validation_data, 
        train_and_validation_labels,
        #validation_split = 0.1, 
        class_weight={0: factor_NSA, 1: 1.0},
        sample_weight=np.reshape(sample_weights, (len(sample_weights), )),
        epochs=epochs,
        batch_size = best_batch_size,
        callbacks=callbacks,
        verbose=1
    ) 
                    
    print("Test %d Iteration %d best params: conv: %d num_cells: %d kernel_size: %d batch_size: %d dropout: %.2f" % (test_number, iteration, best_conv, best_numcells, best_kernel, best_batch_size, best_dropout))
    
    # Output accuracy, validation accuracy, loss and validation loss for all models
    decorate_stats_final(history) 

    # Plot the history
    plt_model_final(SEQ_MODEL_DATA_PATH, test_number, history, 'rnn_model_'+model_name+'_final_model_iteration'+str(iteration))
    
    best_model_file = SEQ_MODEL_DATA_PATH+str(test_number)+'_rnn_model_multiple_properties_final_model_iteration'+str(iteration)+'.h5' 
    
    alpha1, data, labels, sample_weights = boost_seq(best_batch_size, data, labels, best_model_file, sample_weights, 0.5)
    
    models.append(model)
    alpha_values.append(alpha1)
    model_pred_single = model_predict_seq(best_batch_size, test_data, test_labels, '', model) 
    model_predictions.append(model_pred_single)
    model_pred_multiple = multiple_model_predict_seq(0.5, best_batch_size, test_data, test_labels, models, model_predictions, alpha_values)
    adaboost_generate_predictions_seq(0.5, model_pred_single, test_number, "_weak" + str(iteration), test_labels, properties, names, offset, mask_value)

    adaboost_generate_predictions_seq(0.5, model_pred_multiple, test_number, "_iteration" + str(iteration), test_labels, properties, names, offset, mask_value)
    
    if iteration < MAX_ITERATIONS:
        final_train_seq(models, model_predictions, alpha_values, sample_weights, iteration + 1, factor_NSA, epochs, test_number, model_name, data, labels, best_batch_size, best_dropout, best_conv, best_numcells, best_kernel, test_data, test_labels, properties, names, offset, mask_value)
    
def common_boost(train_and_validation_data, batch_size, data, labels, model_file, sample_weights, thr):
    # Load the best model.
    best_model = tf.keras.models.load_model(model_file)    
    # Get model predictions on the data.
    model_predictions = best_model.predict(train_and_validation_data, batch_size=batch_size) 
    model_predictions = convert_list(model_predictions)
    # Calculate error rate base on misclassified samples
    predicted_classes = []
    e1 = 0
    for i in range(len(model_predictions)):
        if model_predictions[i] >= thr:
            predicted_classes.append(1)
        else:
            predicted_classes.append(0)
        if labels[i] != predicted_classes[i]:
            e1 += sample_weights[i] 
    if e1 <= 0:
        e1 = pow(10, -18)
    if e1 >= 1:
        e1 = 1 - pow(10, -18)
    alpha1 = 0.5 * np.log((1 - e1) / e1)
    # Updated weights
    for i in range(len(model_predictions)): 
        if labels[i] != predicted_classes[i]:
            sample_weights[i] = sample_weights[i] * np.exp(alpha1)
        else:
            sample_weights[i] = sample_weights[i] * np.exp(-alpha1)
    sample_weights = sample_weights / sum(sample_weights) 
    return alpha1, data, labels, sample_weights

def boost_AP(num_props, batch_size, data, labels, model_file, sample_weights, thr):
    train_and_validation_data, train_and_validation_labels = reshape_AP(num_props, data, labels)
    return common_boost(train_and_validation_data, batch_size, data, labels, model_file, sample_weights, thr)

def boost_seq(batch_size, data, labels, model_file, sample_weights, thr):
    train_and_validation_data, train_and_validation_labels = reshape_seq(data, labels)
    return common_boost(train_and_validation_data, batch_size, data, labels, model_file, sample_weights, thr)

def boost(num_props, batch_size, data, labels, model_file, sample_weights, thr):
    train_and_validation_data, train_and_validation_labels = reshape(num_props, data, labels)
    return common_boost(train_and_validation_data, batch_size, data, labels, model_file, sample_weights, thr)

def select(sample_weights):
    cumulative = np.cumsum(sample_weights)
    indices = []
    for i in range(len(sample_weights)):
        n = random.random()
        index = 0
        while n > cumulative[index] and index < len(sample_weights) - 1:
            index += 1
        indices.append(index) 
    return indices