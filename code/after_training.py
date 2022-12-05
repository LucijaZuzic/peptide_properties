from custom_plots import convert_to_binary
from utils import MODEL_DATA_PATH, DATA_PATH, SEQ_MODEL_DATA_PATH, MY_MODEL_DATA_PATH
import numpy as np
import pandas as pd
import tensorflow as tf
from automate_training import reshape, reshape_seq, reshape_AP, load_data_SA, merge_data, merge_data_seq, load_data_SA_seq, merge_data_AP, load_data_SA_AP

def no_file_model_predict_seq(best_batch_size, test_data, test_labels, models, alpha_values):
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

def no_file_model_predict(num_props, best_batch_size, test_data, test_labels, models, alpha_values):
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

def no_file_model_predict_AP(num_props, best_batch_size, test_data, test_labels, models, alpha_values):
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
    model_predictions_amino = model_predict_seq(best_batch_size, all_data, all_labels, best_model_file)
    # Write SA probability to file
    
    percentage_filename = SEQ_MODEL_DATA_PATH+str(test_number)+"_"+"percentage_multiple_properties.csv"
    percentage_file = open(percentage_filename, "w", encoding="utf-8")
    percentage_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    for x in range(len(sequences)):
        percentage_string_to_write += sequences[x]+";"+str(np.round(model_predictions_amino[x] * 100, 2))+";"+past_grades[x]+"\n"
    percentage_string_to_write = percentage_string_to_write.replace('.',',')
    percentage_file.write(percentage_string_to_write)
    percentage_file.close()

    # Write class based on the threshold of probability to file
    
    threshold_amino = 0.5 
    
    model_predictions_amino = convert_to_binary(model_predictions_amino, threshold_amino) 
    
    grade_filename = SEQ_MODEL_DATA_PATH+str(test_number)+"_"+"grade_multiple_properties.csv"
    grade_file = open(grade_filename, "w", encoding="utf-8")
    grade_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    correct_amino = 0 
    for x in range(len(sequences)): 
        if (model_predictions_amino[x] == 1 and past_classes[x] == 'Y') or (model_predictions_amino[x] == 0 and past_classes[x] == 'N'):
            correct_amino += 1 
            
        part1 = sequences[x]+";"+str(model_predictions_amino[x])+";"+past_classes[x]+"\n"
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
    model_predictions_amino = model_predict(num_props, best_batch_size, all_data, all_labels, best_model_file)
    # Write SA probability to file
    
    percentage_filename = MODEL_DATA_PATH+str(test_number)+"_"+"percentage_multiple_properties.csv"
    percentage_file = open(percentage_filename, "w", encoding="utf-8")
    percentage_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    for x in range(len(sequences)):
        percentage_string_to_write += sequences[x]+";"+str(np.round(model_predictions_amino[x] * 100, 2))+";"+past_grades[x]+"\n"
    percentage_string_to_write = percentage_string_to_write.replace('.',',')
    percentage_file.write(percentage_string_to_write)
    percentage_file.close()

    # Write class based on the threshold of probability to file
    
    threshold_amino = 0.5 
    
    model_predictions_amino = convert_to_binary(model_predictions_amino, threshold_amino) 
    
    grade_filename = MODEL_DATA_PATH+str(test_number)+"_"+"grade_multiple_properties.csv"
    grade_file = open(grade_filename, "w", encoding="utf-8")
    grade_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    correct_amino = 0 
    for x in range(len(sequences)): 
        if (model_predictions_amino[x] == 1 and past_classes[x] == 'Y') or (model_predictions_amino[x] == 0 and past_classes[x] == 'N'):
            correct_amino += 1 
            
        part1 = sequences[x]+";"+str(model_predictions_amino[x])+";"+past_classes[x]+"\n"
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
    model_predictions_amino = model_predict_AP(num_props, best_batch_size, all_data, all_labels, best_model_file)
    # Write SA probability to file
    
    percentage_filename = MY_MODEL_DATA_PATH+str(test_number)+"_"+"percentage_multiple_properties.csv"
    percentage_file = open(percentage_filename, "w", encoding="utf-8")
    percentage_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    for x in range(len(sequences)):
        percentage_string_to_write += sequences[x]+";"+str(np.round(model_predictions_amino[x] * 100, 2))+";"+past_grades[x]+"\n"
    percentage_string_to_write = percentage_string_to_write.replace('.',',')
    percentage_file.write(percentage_string_to_write)
    percentage_file.close()

    # Write class based on the threshold of probability to file
    
    threshold_amino = 0.5 
    
    model_predictions_amino = convert_to_binary(model_predictions_amino, threshold_amino) 
    
    grade_filename = MY_MODEL_DATA_PATH+str(test_number)+"_"+"grade_multiple_properties.csv"
    grade_file = open(grade_filename, "w", encoding="utf-8")
    grade_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    correct_amino = 0 
    for x in range(len(sequences)): 
        if (model_predictions_amino[x] == 1 and past_classes[x] == 'Y') or (model_predictions_amino[x] == 0 and past_classes[x] == 'N'):
            correct_amino += 1 
            
        part1 = sequences[x]+";"+str(model_predictions_amino[x])+";"+past_classes[x]+"\n"
        part1 = part1.replace(".0",'')
        part1 = part1.replace('1','Y')
        part1 = part1.replace('0','N') 
        grade_string_to_write += part1  
    last_line = "Number of matches with method without RNN;"+str(correct_amino)+";\n"
    last_line = last_line.replace(".",',') 
    grade_string_to_write += last_line
    grade_file.write(grade_string_to_write)
    grade_file.close() 

def no_file_after_training_seq(best_batch_size, test_number, iteration, models, alpha_values, properties, names=['AP'], offset = 1, masking_value=2):
        
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
    model_predictions_amino = no_file_model_predict_seq(best_batch_size, all_data, all_labels, models, alpha_values)
    # Write SA probability to file
    
    percentage_filename = SEQ_MODEL_DATA_PATH+str(test_number)+"_"+"percentage_multiple_properties" + iteration + ".csv"
    percentage_file = open(percentage_filename, "w", encoding="utf-8")
    percentage_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    for x in range(len(sequences)):
        percentage_string_to_write += sequences[x]+";"+str(np.round(model_predictions_amino[x] * 100, 2))+";"+past_grades[x]+"\n"
    percentage_string_to_write = percentage_string_to_write.replace('.',',')
    percentage_file.write(percentage_string_to_write)
    percentage_file.close()

    # Write class based on the threshold of probability to file
    
    threshold_amino = 0.5 
    
    model_predictions_amino = convert_to_binary(model_predictions_amino, threshold_amino) 
    
    grade_filename = SEQ_MODEL_DATA_PATH+str(test_number)+"_"+"grade_multiple_properties" + iteration + ".csv"
    grade_file = open(grade_filename, "w", encoding="utf-8")
    grade_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    correct_amino = 0 
    for x in range(len(sequences)): 
        if (model_predictions_amino[x] == 1 and past_classes[x] == 'Y') or (model_predictions_amino[x] == 0 and past_classes[x] == 'N'):
            correct_amino += 1 
            
        part1 = sequences[x]+";"+str(model_predictions_amino[x])+";"+past_classes[x]+"\n"
        part1 = part1.replace(".0",'')
        part1 = part1.replace('1','Y')
        part1 = part1.replace('0','N') 
        grade_string_to_write += part1  
    last_line = "Number of matches with method without RNN;"+str(correct_amino)+";\n"
    last_line = last_line.replace(".",',') 
    grade_string_to_write += last_line
    grade_file.write(grade_string_to_write)
    grade_file.close() 

def no_file_after_training(num_props, best_batch_size, test_number, iteration, models, alpha_values, properties, names=['AP'], offset = 1, masking_value=2):
        
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
    model_predictions_amino = no_file_model_predict(num_props, best_batch_size, all_data, all_labels, models, alpha_values)
    # Write SA probability to file
    
    percentage_filename = MODEL_DATA_PATH+str(test_number)+"_"+"percentage_multiple_properties" + iteration + ".csv"
    percentage_file = open(percentage_filename, "w", encoding="utf-8")
    percentage_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    for x in range(len(sequences)):
        percentage_string_to_write += sequences[x]+";"+str(np.round(model_predictions_amino[x] * 100, 2))+";"+past_grades[x]+"\n"
    percentage_string_to_write = percentage_string_to_write.replace('.',',')
    percentage_file.write(percentage_string_to_write)
    percentage_file.close()

    # Write class based on the threshold of probability to file
    
    threshold_amino = 0.5 
    
    model_predictions_amino = convert_to_binary(model_predictions_amino, threshold_amino) 
    
    grade_filename = MODEL_DATA_PATH+str(test_number)+"_"+"grade_multiple_properties" + iteration + ".csv"
    grade_file = open(grade_filename, "w", encoding="utf-8")
    grade_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    correct_amino = 0 
    for x in range(len(sequences)): 
        if (model_predictions_amino[x] == 1 and past_classes[x] == 'Y') or (model_predictions_amino[x] == 0 and past_classes[x] == 'N'):
            correct_amino += 1 
            
        part1 = sequences[x]+";"+str(model_predictions_amino[x])+";"+past_classes[x]+"\n"
        part1 = part1.replace(".0",'')
        part1 = part1.replace('1','Y')
        part1 = part1.replace('0','N') 
        grade_string_to_write += part1  
    last_line = "Number of matches with method without RNN;"+str(correct_amino)+";\n"
    last_line = last_line.replace(".",',') 
    grade_string_to_write += last_line
    grade_file.write(grade_string_to_write)
    grade_file.close() 
        
def no_file_after_training_AP(num_props, best_batch_size, test_number, iteration, models, alpha_values, properties, names=['AP'], offset = 1, masking_value=2):
        
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
    model_predictions_amino = no_file_model_predict_AP(num_props, best_batch_size, all_data, all_labels, models, alpha_values)
    # Write SA probability to file
    
    percentage_filename = MY_MODEL_DATA_PATH+str(test_number)+"_"+"percentage_multiple_properties" + iteration + ".csv"
    percentage_file = open(percentage_filename, "w", encoding="utf-8")
    percentage_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    for x in range(len(sequences)):
        percentage_string_to_write += sequences[x]+";"+str(np.round(model_predictions_amino[x] * 100, 2))+";"+past_grades[x]+"\n"
    percentage_string_to_write = percentage_string_to_write.replace('.',',')
    percentage_file.write(percentage_string_to_write)
    percentage_file.close()

    # Write class based on the threshold of probability to file
    
    threshold_amino = 0.5 
    
    model_predictions_amino = convert_to_binary(model_predictions_amino, threshold_amino) 
    
    grade_filename = MY_MODEL_DATA_PATH+str(test_number)+"_"+"grade_multiple_properties" + iteration + ".csv"
    grade_file = open(grade_filename, "w", encoding="utf-8")
    grade_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"
    
    correct_amino = 0 
    for x in range(len(sequences)): 
        if (model_predictions_amino[x] == 1 and past_classes[x] == 'Y') or (model_predictions_amino[x] == 0 and past_classes[x] == 'N'):
            correct_amino += 1 
            
        part1 = sequences[x]+";"+str(model_predictions_amino[x])+";"+past_classes[x]+"\n"
        part1 = part1.replace(".0",'')
        part1 = part1.replace('1','Y')
        part1 = part1.replace('0','N') 
        grade_string_to_write += part1  
    last_line = "Number of matches with method without RNN;"+str(correct_amino)+";\n"
    last_line = last_line.replace(".",',') 
    grade_string_to_write += last_line
    grade_file.write(grade_string_to_write)
    grade_file.close() 