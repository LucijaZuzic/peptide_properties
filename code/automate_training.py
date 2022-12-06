import tensorflow as tf
import numpy as np
import pandas as pd
from custom_plots import plt_model, plt_model_final, decorate_stats, decorate_stats_avg, decorate_stats_final
from utils import DATA_PATH, MODEL_DATA_PATH, SEQ_MODEL_DATA_PATH, MY_MODEL_DATA_PATH, scale, split_amino_acids, split_dipeptides, split_tripeptides, padding
import new_model
from sklearn.preprocessing import MinMaxScaler 
from seqprops import SequentialPropertiesEncoder 
import random
from custom_plots import convert_to_binary, make_ROC_plots, make_PR_plots, output_metrics, hist_predicted

MAX_ITERATIONS = 5
 
def no_file_model_predict_seq(best_batch_size, test_data, test_labels, models, model_predictions, alpha_values):
    test_data, test_labels = reshape_seq(test_data, test_labels)
    model_predictions = []
    
    # Get model predictions on the test data.
    for i in range(len(models)):
        tmp_model_predictions = models[i].predict(test_data, batch_size=best_batch_size) * alpha_values[i] / sum(alpha_values)
        
        if len(model_predictions) == 0:
            for j in range(len(tmp_model_predictions)):
                model_predictions.append(tmp_model_predictions[j])
        else:
            for j in range(len(tmp_model_predictions)):
                model_predictions[j] += tmp_model_predictions[j]

    return model_predictions 

def no_file_model_predict(num_props, best_batch_size, test_data, test_labels, models, model_predictions, alpha_values):
    test_data, test_labels = reshape(num_props, test_data, test_labels) 
    model_predictions = []
    
    # Get model predictions on the test data.
    for i in range(len(models)):
        tmp_model_predictions = models[i].predict(test_data, batch_size=best_batch_size) * alpha_values[i] / sum(alpha_values)
        
        if len(model_predictions) == 0:
            for j in range(len(tmp_model_predictions)):
                model_predictions.append(tmp_model_predictions[j])
        else:
            for j in range(len(tmp_model_predictions)):
                model_predictions[j] += tmp_model_predictions[j]

    return model_predictions 

def no_file_model_predict_AP(num_props, best_batch_size, test_data, test_labels, models, model_predictions, alpha_values):
    test_data, test_labels = reshape_AP(num_props, test_data, test_labels) 
    model_predictions = []
    
    # Get model predictions on the test data.
    for i in range(len(models)):
        tmp_model_predictions = models[i].predict(test_data, batch_size=best_batch_size) * alpha_values[i] / sum(alpha_values)
        
        if len(model_predictions) == 0:
            for j in range(len(tmp_model_predictions)):
                model_predictions.append(tmp_model_predictions[j])
        else:
            for j in range(len(tmp_model_predictions)):
                model_predictions[j] += tmp_model_predictions[j]

    return model_predictions 

def model_predict_seq(best_batch_size, test_data, test_labels, best_model_file):
    # Load the best model.
    best_model = tf.keras.models.load_model(best_model_file) 
    test_data, test_labels = reshape_seq(test_data, test_labels)
    
    # Get model predictions on the test data.
    model_predictions = best_model.predict(test_data, batch_size=best_batch_size)

    return model_predictions 

def model_predict(num_props, best_batch_size, test_data, test_labels, best_model_file):
    # Load the best model.
    best_model = tf.keras.models.load_model(best_model_file)   
    test_data, test_labels = reshape(num_props, test_data, test_labels) 
    # Get model predictions on the test data.
    
    model_predictions = best_model.predict(test_data, batch_size=best_batch_size)

    return model_predictions 

def model_predict_AP(num_props, best_batch_size, test_data, test_labels, best_model_file):
    # Load the best model.
    best_model = tf.keras.models.load_model(best_model_file)   
    test_data, test_labels = reshape_AP(num_props, test_data, test_labels) 
    # Get model predictions on the test data.
    
    model_predictions = best_model.predict(test_data, batch_size=best_batch_size)

    return model_predictions 

def after_training_seq(best_batch_size, test_number, best_model_file, properties, names=['AP'], offset = 1, masking_value=2):
        
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
    
    SA_data = []
    for i in range(len(sequences)):
        SA_data.append([sequences[i], '0'])
         
    SA, NSA = load_data_SA_seq(SA_data, names, offset, properties)
    all_data, all_labels = merge_data_seq(SA, NSA)
    all_data = all_data[:-1] 
    all_labels = all_labels[:-1]  
    sequences = sequences[:-1] 
      
    # Generate predictions on data that has no label beforehand 
    model_predictions = model_predict_seq(best_batch_size, all_data, all_labels, best_model_file)
    # Write SA probability to file
    
    percentage_filename = SEQ_MODEL_DATA_PATH+str(test_number)+"_"+"percentage_multiple_properties.csv"
    percentage_file = open(percentage_filename, "w", encoding="utf-8")
    percentage_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    for x in range(len(sequences)):
        percentage_string_to_write += sequences[x]+";"+str(np.round(model_predictions[x] * 100, 2))+";"+past_grades[x]+"\n"
    percentage_string_to_write = percentage_string_to_write.replace('.',',')
    percentage_file.write(percentage_string_to_write)
    percentage_file.close()

    # Write class based on the threshold of probability to file
    
    threshold_amino = 0.5 
    
    model_predictions = convert_to_binary(model_predictions, threshold_amino) 
    
    grade_filename = SEQ_MODEL_DATA_PATH+str(test_number)+"_"+"grade_multiple_properties.csv"
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

def after_training(num_props, best_batch_size, test_number, best_model_file, properties, names=['AP'], offset = 1, masking_value=2):
        
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
    
    SA_data = []
    for i in range(len(sequences)):
        SA_data.append([sequences[i], '0'])
         
    SA, NSA = load_data_SA(SA_data, names, offset, properties)
    all_data, all_labels = merge_data(SA, NSA)
    all_data = all_data[:-1] 
    all_labels = all_labels[:-1] 
    sequences = sequences[:-1] 
      
    # Generate predictions on data that has no label beforehand 
    model_predictions = model_predict(num_props, best_batch_size, all_data, all_labels, best_model_file)
    # Write SA probability to file
    
    percentage_filename = MODEL_DATA_PATH+str(test_number)+"_"+"percentage_multiple_properties.csv"
    percentage_file = open(percentage_filename, "w", encoding="utf-8")
    percentage_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    for x in range(len(sequences)):
        percentage_string_to_write += sequences[x]+";"+str(np.round(model_predictions[x] * 100, 2))+";"+past_grades[x]+"\n"
    percentage_string_to_write = percentage_string_to_write.replace('.',',')
    percentage_file.write(percentage_string_to_write)
    percentage_file.close()

    # Write class based on the threshold of probability to file
    
    threshold_amino = 0.5 
    
    model_predictions = convert_to_binary(model_predictions, threshold_amino) 
    
    grade_filename = MODEL_DATA_PATH+str(test_number)+"_"+"grade_multiple_properties.csv"
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
        
def after_training_AP(num_props, best_batch_size, test_number, best_model_file, properties, names=['AP'], offset = 1, masking_value=2):
        
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
    
    SA_data = []
    for i in range(len(sequences)):
        SA_data.append([sequences[i], '0'])
         
    SA, NSA = load_data_SA_AP(SA_data, names, offset, properties)
    all_data, all_labels = merge_data_AP(SA, NSA)
    all_data = all_data[:-1] 
    all_labels = all_labels[:-1] 
    sequences = sequences[:-1] 
      
    # Generate predictions on data that has no label beforehand 
    model_predictions = model_predict_AP(num_props, best_batch_size, all_data, all_labels, best_model_file)
    # Write SA probability to file
    
    percentage_filename = MY_MODEL_DATA_PATH+str(test_number)+"_"+"percentage_multiple_properties.csv"
    percentage_file = open(percentage_filename, "w", encoding="utf-8")
    percentage_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    for x in range(len(sequences)):
        percentage_string_to_write += sequences[x]+";"+str(np.round(model_predictions[x] * 100, 2))+";"+past_grades[x]+"\n"
    percentage_string_to_write = percentage_string_to_write.replace('.',',')
    percentage_file.write(percentage_string_to_write)
    percentage_file.close()

    # Write class based on the threshold of probability to file
    
    threshold_amino = 0.5 
    
    model_predictions = convert_to_binary(model_predictions, threshold_amino) 
    
    grade_filename = MY_MODEL_DATA_PATH+str(test_number)+"_"+"grade_multiple_properties.csv"
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

def no_file_after_training_seq(best_batch_size, test_number, iteration, models, model_predictions, alpha_values, properties, names=['AP'], offset = 1):
        
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
    
    SA_data = []
    for i in range(len(sequences)):
        SA_data.append([sequences[i], '0'])
         
    SA, NSA = load_data_SA_seq(SA_data, names, offset, properties)
    all_data, all_labels = merge_data_seq(SA, NSA)
    all_data = all_data[:-1] 
    all_labels = all_labels[:-1]  
    sequences = sequences[:-1] 
      
    # Generate predictions on data that has no label beforehand 
    if len(model_predictions) == 0:
        model_predictions = no_file_model_predict_seq(best_batch_size, all_data, all_labels, models, model_predictions, alpha_values)

    # Write SA probability to file
    percentage_filename = SEQ_MODEL_DATA_PATH+str(test_number)+"_"+"percentage_multiple_properties" + iteration + ".csv"
    percentage_file = open(percentage_filename, "w", encoding="utf-8")
    percentage_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    for x in range(len(sequences)):
        percentage_string_to_write += sequences[x]+";"+str(np.round(model_predictions[x] * 100, 2))+";"+past_grades[x]+"\n"
    percentage_string_to_write = percentage_string_to_write.replace('.',',')
    percentage_file.write(percentage_string_to_write)
    percentage_file.close()

    # Write class based on the threshold of probability to file
    threshold_amino = 0.5 
    
    model_predictions = convert_to_binary(model_predictions, threshold_amino) 
    
    grade_filename = SEQ_MODEL_DATA_PATH+str(test_number)+"_"+"grade_multiple_properties" + iteration + ".csv"
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

def no_file_after_training(num_props, best_batch_size, test_number, iteration, models, model_predictions, alpha_values, properties, names=['AP'], offset = 1):
        
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
    
    SA_data = []
    for i in range(len(sequences)):
        SA_data.append([sequences[i], '0'])
         
    SA, NSA = load_data_SA(SA_data, names, offset, properties)
    all_data, all_labels = merge_data(SA, NSA)
    all_data = all_data[:-1] 
    all_labels = all_labels[:-1] 
    sequences = sequences[:-1] 
      
    # Generate predictions on data that has no label beforehand 
    if len(model_predictions) == 0:
        model_predictions = no_file_model_predict(num_props, best_batch_size, all_data, all_labels, models, model_predictions, alpha_values)

    # Write SA probability to file
    percentage_filename = MODEL_DATA_PATH+str(test_number)+"_"+"percentage_multiple_properties" + iteration + ".csv"
    percentage_file = open(percentage_filename, "w", encoding="utf-8")
    percentage_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    for x in range(len(sequences)):
        percentage_string_to_write += sequences[x]+";"+str(np.round(model_predictions[x] * 100, 2))+";"+past_grades[x]+"\n"
    percentage_string_to_write = percentage_string_to_write.replace('.',',')
    percentage_file.write(percentage_string_to_write)
    percentage_file.close()

    # Write class based on the threshold of probability to file
    threshold_amino = 0.5 
    
    model_predictions = convert_to_binary(model_predictions, threshold_amino) 
    
    grade_filename = MODEL_DATA_PATH+str(test_number)+"_"+"grade_multiple_properties" + iteration + ".csv"
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
        
def no_file_after_training_AP(num_props, best_batch_size, test_number, iteration, models, model_predictions, alpha_values, properties, names=['AP'], offset = 1):
        
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
    
    SA_data = []
    for i in range(len(sequences)):
        SA_data.append([sequences[i], '0'])
         
    SA, NSA = load_data_SA_AP(SA_data, names, offset, properties)
    all_data, all_labels = merge_data_AP(SA, NSA)
    all_data = all_data[:-1] 
    all_labels = all_labels[:-1] 
    sequences = sequences[:-1] 
      
    # Generate predictions on data that has no label beforehand 
    if len(model_predictions) == 0:
        model_predictions = no_file_model_predict_AP(num_props, best_batch_size, all_data, all_labels, models, model_predictions, alpha_values)

    # Write SA probability to file
    percentage_filename = MY_MODEL_DATA_PATH+str(test_number)+"_"+"percentage_multiple_properties" + iteration + ".csv"
    percentage_file = open(percentage_filename, "w", encoding="utf-8")
    percentage_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    for x in range(len(sequences)):
        percentage_string_to_write += sequences[x]+";"+str(np.round(model_predictions[x] * 100, 2))+";"+past_grades[x]+"\n"
    percentage_string_to_write = percentage_string_to_write.replace('.',',')
    percentage_file.write(percentage_string_to_write)
    percentage_file.close()

    # Write class based on the threshold of probability to file
    threshold_amino = 0.5 
    
    model_predictions = convert_to_binary(model_predictions, threshold_amino) 
    
    grade_filename = MY_MODEL_DATA_PATH+str(test_number)+"_"+"grade_multiple_properties" + iteration + ".csv"
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

def generate_predictions(num_props, best_batch_size, best_model_file, test_number, test_data, test_labels, properties, names = ['AP'], offset = 1, masking_value = 2): 
     # Get predictions from all the models for data that was labeled beforehand
     model_predictions = model_predict(num_props, best_batch_size, test_data, test_labels, best_model_file) 
      
     #Plot ROC curves for all models
     make_ROC_plots(MODEL_DATA_PATH, test_number, test_labels, model_predictions=model_predictions)
     
     #Plot PR curves for all models
     make_PR_plots(MODEL_DATA_PATH, test_number, test_labels, model_predictions=model_predictions)
    
     # Output adjusted accuracy, F1 score and ROC AUC score for all models
     output_metrics(test_labels, model_predictions) 
     
     # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
     hist_predicted(MODEL_DATA_PATH, test_number, test_labels, model_predictions) 
    
     # Generate predictions on data that has no label beforehand
     after_training(num_props, best_batch_size, test_number, best_model_file, properties, names, offset, masking_value)
     
def generate_predictions_seq(best_batch_size, best_model_file, test_number, test_data, test_labels, properties, names = ['AP'], offset = 1, masking_value = 2): 
     # Get predictions from all the models for data that was labeled beforehand
     model_predictions = model_predict_seq(best_batch_size, test_data, test_labels, best_model_file) 
      
     #Plot ROC curves for all models
     make_ROC_plots(SEQ_MODEL_DATA_PATH, test_number, test_labels, model_predictions=model_predictions)
     
     #Plot PR curves for all models
     make_PR_plots(SEQ_MODEL_DATA_PATH, test_number, test_labels, model_predictions=model_predictions)
    
     # Output adjusted accuracy, F1 score and ROC AUC score for all models
     output_metrics(test_labels, model_predictions) 
     
     # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
     hist_predicted(SEQ_MODEL_DATA_PATH, test_number, test_labels, model_predictions) 
    
     # Generate predictions on data that has no label beforehand
     after_training_seq(best_batch_size, test_number, best_model_file, properties, names, offset, masking_value)

def generate_predictions_AP(num_props, best_batch_size, best_model_file, test_number, test_data, test_labels, properties, names = ['AP'], offset = 1, masking_value = 2): 
    # Get predictions from all the models for data that was labeled beforehand
    model_predictions = model_predict_AP(num_props, best_batch_size, test_data, test_labels, best_model_file) 
     
    #Plot ROC curves for all models
    make_ROC_plots(MY_MODEL_DATA_PATH, test_number, test_labels, model_predictions=model_predictions)
    
    #Plot PR curves for all models
    make_PR_plots(MY_MODEL_DATA_PATH, test_number, test_labels, model_predictions=model_predictions)
   
    # Output adjusted accuracy, F1 score and ROC AUC score for all models
    output_metrics(test_labels, model_predictions) 
    
    # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
    hist_predicted(MY_MODEL_DATA_PATH, test_number, test_labels, model_predictions) 
   
    # Generate predictions on data that has no label beforehand
    after_training_AP(num_props, best_batch_size, test_number, best_model_file, properties, names, offset, masking_value)
    
def adaboost_generate_predictions(models, model_predictions, alpha_values, num_props, best_batch_size, test_number, iteration, test_data, test_labels, properties, names = ['AP'], offset = 1): 
     # Get predictions from all the models for data that was labeled beforehand
     if len(model_predictions) == 0:
        model_predictions = no_file_model_predict(num_props, best_batch_size, test_data, test_labels, models, model_predictions, alpha_values) 
      
     #Plot ROC curves for all models
     make_ROC_plots(MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions=model_predictions)
     
     #Plot PR curves for all models
     make_PR_plots(MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions=model_predictions)
    
     # Output adjusted accuracy, F1 score and ROC AUC score for all models
     output_metrics(test_labels, model_predictions) 
     
     # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
     hist_predicted(MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions) 
    
     # Generate predictions on data that has no label beforehand
     no_file_after_training(num_props, best_batch_size, test_number, iteration, models, model_predictions, alpha_values, properties, names, offset)
     
def adaboost_generate_predictions_seq(models, model_predictions, alpha_values, best_batch_size, test_number, iteration, test_data, test_labels, properties, names = ['AP'], offset = 1):  
     # Get predictions from all the models for data that was labeled beforehand
     if len(model_predictions) == 0:   
        model_predictions = no_file_model_predict_seq(best_batch_size, test_data, test_labels, models, model_predictions, alpha_values) 
      
     #Plot ROC curves for all models
     make_ROC_plots(SEQ_MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions=model_predictions)
     
     #Plot PR curves for all models
     make_PR_plots(SEQ_MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions=model_predictions)
    
     # Output adjusted accuracy, F1 score and ROC AUC score for all models
     output_metrics(test_labels, model_predictions) 
     
     # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
     hist_predicted(SEQ_MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions) 
    
     # Generate predictions on data that has no label beforehand
     no_file_after_training_seq(best_batch_size, test_number, iteration, models, model_predictions, alpha_values, properties, names, offset)

def adaboost_generate_predictions_AP(models, model_predictions, alpha_values, num_props, best_batch_size, test_number, iteration, test_data, test_labels, properties, names = ['AP'], offset = 1):  
    # Get predictions from all the models for data that was labeled beforehand
    if len(model_predictions) == 0:    
        model_predictions = no_file_model_predict_AP(num_props, best_batch_size, test_data, test_labels, models, model_predictions, alpha_values) 
     
    #Plot ROC curves for all models
    make_ROC_plots(MY_MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions=model_predictions)
    
    #Plot PR curves for all models
    make_PR_plots(MY_MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions=model_predictions)
   
    # Output adjusted accuracy, F1 score and ROC AUC score for all models
    output_metrics(test_labels, model_predictions) 
    
    # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
    hist_predicted(MY_MODEL_DATA_PATH, test_number, iteration, test_labels, model_predictions) 
   
    # Generate predictions on data that has no label beforehand
    no_file_after_training_AP(num_props, best_batch_size, test_number, iteration, models, model_predictions, alpha_values, properties, names, offset)

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
 
def load_data_SA_seq(SA_data, names=['AP'], offset = 1, properties_to_include = [], masking_value=2):

    sequences = []
    labels = []
    for peptide in SA_data:
        if len(peptide[0]) > 24:
            continue
        sequences.append(peptide[0])
        labels.append(peptide[1])
            
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

    sequences = []
    labels = []
    for peptide in SA_data:
        if len(peptide[0]) > 24:
            continue
        sequences.append(peptide[0])
        labels.append(peptide[1])
            
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

    sequences = []
    labels = []
    maxlen = 24
    for peptide in SA_data:
        if len(peptide[0]) > 24:
            continue
        sequences.append(peptide[0]) 
        labels.append(peptide[1]) 
     
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
                    
            amino_acids_ap_padded = padding(amino_acids_ap, maxlen, masking_value)
            dipeptides_acids_ap_padded = padding(dipeptides_ap, maxlen, masking_value)
            tripeptides_ap_padded = padding(tripeptides_ap, maxlen, masking_value)  
        
            new_props.append(amino_acids_ap_padded)
            new_props.append(dipeptides_acids_ap_padded)
            new_props.append(tripeptides_ap_padded) 
        
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
    #hyperparameter_numcells = [32]
    #hyperparameter_kernel_size = [8]
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
                            
                            callbacks = [
                                # When validation loss doesn't decrease in 10 consecutive epochs, reduce the learning rate by 90%.
                                # This is repeated while learning rate is >= 0.000001.
                                tf.keras.callbacks.ReduceLROnPlateau(
                                    monitor='val_loss', factor=0.1, patience=10, min_lr=0.000001
                                ),
                                # Save the best model (the one with the lowest validation loss).
                                tf.keras.callbacks.ModelCheckpoint(
                                    model_file, save_best_only=True, monitor='val_loss', mode='min'
                                ),
                                # This callback will stop the training when there is no improvement in
                                # the loss for three consecutive epochs.
                                # Restoring best weights in case of performance drop
                                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                            ]
                    
                            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                    
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
            
    models, model_predictions, alpha_values = final_train_seq([], [], [], sample_weights, 1, factor_NSA, epochs, test_number, model_name, train_and_validation_data, train_and_validation_labels, best_batch_size, best_dropout, best_conv, best_numcells, best_kernel, test_data, test_labels, properties, names, offset, mask_value)
    
    adaboost_generate_predictions_seq(models, model_predictions, alpha_values, best_batch_size, test_number, "_final", test_data, test_labels, properties, names, offset)

def model_training(num_props, test_number, train_and_validation_data, train_and_validation_labels, kfold_second, epochs, factor_NSA, test_data, test_labels, properties, names, offset, mask_value=2):
    
    model_name = "multiple_properties" 
           
    params_nr = 0 
    min_val_loss = 1000
    
    hyperparameter_conv = [5]
    hyperparameter_numcells = [32, 48, 64]
    hyperparameter_kernel_size = [4, 6, 8]
    #hyperparameter_numcells = [32]
    #hyperparameter_kernel_size = [8]
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
                                        
                                        callbacks = [
                                            # When validation loss doesn't decrease in 10 consecutive epochs, reduce the learning rate by 90%.
                                            # This is repeated while learning rate is >= 0.000001.
                                            tf.keras.callbacks.ReduceLROnPlateau(
                                                monitor='val_loss', factor=0.1, patience=10, min_lr=0.000001
                                            ),
                                            # Save the best model (the one with the lowest validation loss).
                                            tf.keras.callbacks.ModelCheckpoint(
                                                model_file, save_best_only=True, monitor='val_loss', mode='min'
                                            ),
                                            # This callback will stop the training when there is no improvement in
                                            # the loss for three consecutive epochs.
                                            # Restoring best weights in case of performance drop
                                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                                        ]
                                
                                        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                                
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
            
    models, model_predictions, alpha_values = final_train([], [], [], sample_weights, 1, factor_NSA, epochs, test_number, model_name, num_props, train_and_validation_data, train_and_validation_labels, best_batch_size, best_lstm, best_dense, best_dropout, best_lambda, best_conv, best_numcells, best_kernel, test_data, test_labels, properties, names, offset, mask_value)

    adaboost_generate_predictions(models, model_predictions, alpha_values, num_props, best_batch_size, test_number, "_final", test_data, test_labels, properties, names, offset)

def model_training_AP(num_props, test_number, train_and_validation_data, train_and_validation_labels, kfold_second, epochs, factor_NSA, test_data, test_labels, properties, names, offset, mask_value=2):

    model_name = "multiple_properties" 
           
    params_nr = 0 
    min_val_loss = 1000
      
    hyperparameter_lstm = [5]
    hyperparameter_dense = [64, 96, 128] 
    #hyperparameter_dense = [128] 
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
                            
                            callbacks = [
                                # When validation loss doesn't decrease in 10 consecutive epochs, reduce the learning rate by 90%.
                                # This is repeated while learning rate is >= 0.000001.
                                tf.keras.callbacks.ReduceLROnPlateau(
                                    monitor='val_loss', factor=0.1, patience=10, min_lr=0.000001
                                ),
                                # Save the best model (the one with the lowest validation loss).
                                tf.keras.callbacks.ModelCheckpoint(
                                    model_file, save_best_only=True, monitor='val_loss', mode='min'
                                ),
                                # This callback will stop the training when there is no improvement in
                                # the loss for three consecutive epochs.
                                # Restoring best weights in case of performance drop
                                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                            ]
                    
                            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                    
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
            
    models, model_predictions, alpha_values = final_train_AP([], [], [], sample_weights, 1, factor_NSA, epochs, test_number, model_name, num_props, train_and_validation_data, train_and_validation_labels, best_batch_size, best_lstm, best_dense, best_dropout, best_lambda, test_data, test_labels, properties, names, offset, mask_value)
 
    adaboost_generate_predictions_AP(models, model_predictions, alpha_values, num_props, best_batch_size, test_number, "_final", test_data, test_labels, properties, names, offset)

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
    
    callbacks = [
        # When validation loss doesn't decrease in 10 consecutive epochs, reduce the learning rate by 90%.
        # This is repeated while learning rate is >= 0.000001.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=10, min_lr=0.000001
        ),
        # Save the best model (the one with the lowest validation loss).
        tf.keras.callbacks.ModelCheckpoint(
            model_file, save_best_only=True, monitor='loss', mode='min'
        ),
        # This callback will stop the training when there is no improvement in
        # the loss for three consecutive epochs.
        # Restoring best weights in case of performance drop
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    ]

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

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
    model_predictions.append(no_file_after_training_AP(best_batch_size, test_number, "_weak" + str(iteration), [model], [], [alpha1], properties, names, offset))

    adaboost_generate_predictions_AP(models, model_predictions, alpha_values, num_props, best_batch_size, test_number, "_iteration" + str(iteration), test_data, test_labels, properties, names, offset)
    
    if iteration < MAX_ITERATIONS:
        return final_train_AP(models, model_predictions, alpha_values, sample_weights, iteration + 1, factor_NSA, epochs, test_number, model_name, num_props, data, labels, best_batch_size, best_lstm, best_dense, best_dropout, best_lambda, test_data, test_labels, properties, names, offset,  mask_value)
    else:
        return models, model_predictions, alpha_values
   
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
    
    callbacks = [
        # When validation loss doesn't decrease in 10 consecutive epochs, reduce the learning rate by 90%.
        # This is repeated while learning rate is >= 0.000001.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=10, min_lr=0.000001
        ),
        # Save the best model (the one with the lowest validation loss).
        tf.keras.callbacks.ModelCheckpoint(
            model_file, save_best_only=True, monitor='loss', mode='min'
        ),
        # This callback will stop the training when there is no improvement in
        # the loss for three consecutive epochs.
        # Restoring best weights in case of performance drop
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    ]

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

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
    model_predictions.append(no_file_after_training(best_batch_size, test_number, "_weak" + str(iteration), [model], [], [alpha1], properties, names, offset))

    adaboost_generate_predictions(models, model_predictions, alpha_values, num_props, best_batch_size, test_number, "_iteration" + str(iteration), test_data, test_labels, properties, names, offset)

    if iteration < MAX_ITERATIONS:
        return final_train(models, model_predictions, alpha_values, sample_weights, iteration + 1, factor_NSA, epochs, test_number, model_name, num_props, data, labels, best_batch_size, best_lstm, best_dense, best_dropout, best_lambda, best_conv, best_numcells, best_kernel, test_data, test_labels, properties, names, offset,  mask_value)
    else:
        return models, model_predictions, alpha_values

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
    
    callbacks = [
        # When validation loss doesn't decrease in 10 consecutive epochs, reduce the learning rate by 90%.
        # This is repeated while learning rate is >= 0.000001.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=10, min_lr=0.000001
        ),
        # Save the best model (the one with the lowest validation loss).
        tf.keras.callbacks.ModelCheckpoint(
            model_file, save_best_only=True, monitor='loss', mode='min'
        ),
        # This callback will stop the training when there is no improvement in
        # the loss for three consecutive epochs.
        # Restoring best weights in case of performance drop
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    ]

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

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
    model_predictions.append(no_file_after_training_seq(best_batch_size, test_number, "_weak" + str(iteration), [model], [], [alpha1], properties, names, offset))

    adaboost_generate_predictions_seq(models, model_predictions, alpha_values, best_batch_size, test_number, "_iteration" + str(iteration), test_data, test_labels, properties, names, offset)
    
    if iteration < MAX_ITERATIONS:
        return final_train_seq(models, model_predictions, alpha_values, sample_weights, iteration + 1, factor_NSA, epochs, test_number, model_name, data, labels, best_batch_size, best_dropout, best_conv, best_numcells, best_kernel, test_data, test_labels, properties, names, offset, mask_value)
    else:
        return models, model_predictions, alpha_values
    
def boost_AP(num_props, batch_size, data, labels, model_file, sample_weights, thr):
    # Load the best model.
    best_model = tf.keras.models.load_model(model_file)    
    # Get model predictions on the data.
    train_and_validation_data, train_and_validation_labels = reshape_AP(num_props, data, labels)
    model_predictions = best_model.predict(train_and_validation_data, batch_size=batch_size) 
    # Calculate error rate base on misclassified samples
    predicted_classes = []
    e1 = 0
    for i in range(len(model_predictions)):
        for pred in model_predictions[i]: 
            if pred >= thr:
                predicted_classes.append(1)
            else:
                predicted_classes.append(0)
            if labels[i] != predicted_classes[i]:
                e1 += sample_weights[i] 
    alpha1 = 0.5 * np.log((1 - (e1 + pow(10, -12))) / (e1 + pow(10, -12)))
    # Updated weights
    for i in range(len(model_predictions)):
        for pred in model_predictions[i]: 
            if labels[i] != predicted_classes[i]:
                sample_weights[i] = sample_weights[i] * np.exp(alpha1)
            else:
                sample_weights[i] = sample_weights[i] * np.exp(-alpha1)
    sample_weights = sample_weights / sum(sample_weights) 
    # Select new samples based on new weights
    indices = select(sample_weights)
    new_data = []
    new_labels = [] 
    new_weights = [] 
    for index in indices:
        new_data.append(data[index])
        new_labels.append(labels[index])
        new_weights.append(sample_weights[index])
    return alpha1, new_data, new_labels, new_weights
    
def boost_seq(batch_size, data, labels, model_file, sample_weights, thr):
    # Load the best model.
    best_model = tf.keras.models.load_model(model_file)    
    # Get model predictions on the data.
    train_and_validation_data, train_and_validation_labels = reshape_seq(data, labels)
    model_predictions = best_model.predict(train_and_validation_data, batch_size=batch_size) 
    # Calculate error rate base on misclassified samples
    predicted_classes = []
    e1 = 0
    for i in range(len(model_predictions)):
        for pred in model_predictions[i]: 
            if pred >= thr:
                predicted_classes.append(1)
            else:
                predicted_classes.append(0)
            if labels[i] != predicted_classes[i]:
                e1 += sample_weights[i] 
    alpha1 = 0.5 * np.log((1 - (e1 + pow(10, -12))) / (e1 + pow(10, -12)))
    # Updated weights
    for i in range(len(model_predictions)):
        for pred in model_predictions[i]: 
            if labels[i] != predicted_classes[i]:
                sample_weights[i] = sample_weights[i] * np.exp(alpha1)
            else:
                sample_weights[i] = sample_weights[i] * np.exp(-alpha1)
    sample_weights = sample_weights / sum(sample_weights) 
    # Select new samples based on new weights
    indices = select(sample_weights)
    new_data = []
    new_labels = [] 
    new_weights = [] 
    for index in indices:
        new_data.append(data[index])
        new_labels.append(labels[index])
        new_weights.append(sample_weights[index])
    return alpha1, new_data, new_labels, new_weights

def boost(num_props, batch_size, data, labels, model_file, sample_weights, thr):
    # Load the best model.
    best_model = tf.keras.models.load_model(model_file)    
    # Get model predictions on the data.
    train_and_validation_data, train_and_validation_labels = reshape(num_props, data, labels)
    model_predictions = best_model.predict(train_and_validation_data, batch_size=batch_size) 
    # Calculate error rate base on misclassified samples
    predicted_classes = []
    e1 = 0
    for i in range(len(model_predictions)):
        for pred in model_predictions[i]: 
            if pred >= thr:
                predicted_classes.append(1)
            else:
                predicted_classes.append(0)
            if labels[i] != predicted_classes[i]:
                e1 += sample_weights[i] 
    alpha1 = 0.5 * np.log((1 - (e1 + pow(10, -12))) / (e1 + pow(10, -12)))
    # Updated weights
    for i in range(len(model_predictions)):
        for pred in model_predictions[i]: 
            if labels[i] != predicted_classes[i]:
                sample_weights[i] = sample_weights[i] * np.exp(alpha1)
            else:
                sample_weights[i] = sample_weights[i] * np.exp(-alpha1)
    sample_weights = sample_weights / sum(sample_weights) 
    # Select new samples based on new weights
    indices = select(sample_weights)
    new_data = []
    new_labels = [] 
    new_weights = [] 
    for index in indices:
        new_data.append(data[index])
        new_labels.append(labels[index])
        new_weights.append(sample_weights[index])
    return alpha1, new_data, new_labels, new_weights

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