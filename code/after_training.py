from custom_plots import convert_to_binary
import keras
from utils import MODEL_DATA_PATH, MERGE_MODEL_DATA_PATH, DATA_PATH, split_amino_acids, split_dipeptides, split_tripeptides
import numpy as np
from automate_training import load_data_AP, average_model, data_and_labels_from_indices
import pandas as pd 
from data_generator import DataGenerator
from sklearn.preprocessing import MinMaxScaler 
from seqprops import SequentialPropertiesEncoder 

def model_predict(test_data, test_labels, best_model_file, model_type=-1, merge=False, no_model_type = False):
    # Load the best model.
    best_model = keras.models.load_model(best_model_file)

    data_index = model_type
    test_data, test_labels = test_data, test_labels
    if no_model_type == False:
        # Clean the test data from unecessarry data
        test_data, test_labels = data_and_labels_from_indices(test_data, test_labels, list(range(len(test_data))), data_index)
     

    # Check if the model is ansamble or single
    number_items = 1
    if model_type == -1:
        if merge:
            number_items = 6
        else:
            number_items = 3
    elif merge:
        number_items = 2
    if no_model_type == True:
        if model_type == 0:
            number_items = 0
        else:
            number_items = 4

    # Get model predictions on the test data.
    model_predictions = best_model.predict(
        DataGenerator(test_data, test_labels, number_items=number_items, shuffle=False)
    ).flatten() 

    return model_predictions 

def after_training(test_number, best_model_file_amino, best_model_file_di, best_model_file_tri, best_model_file_ansamble, data_to_load1="AP", data_to_load2="logP", merge = False):
    data_to_load = data_to_load1
    if merge:
        data_to_load = data_to_load1 + "_" + data_to_load2  
        
    # Get sequences for peptides with no labels and predictions from the model without machine learning 
    resulteval = DATA_PATH+'RESULTEVAL.csv' 
    df = pd.read_csv(resulteval, skipinitialspace=True, sep=';')
    sequences = list(df['Dizajnirani peptid'])
    past_grades = list(df['Postotak svojstava koja imaju AUC > 0,5 koja su dala SA'])
    past_classes= list(df['SA'])

    new_data_ansamble = []
    labels = np.zeros(len(sequences))

    # Convert each peptide sequence to three lists of AP scores: amino acid, dipeptide and tripeptide scores.
    if not merge:
        amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP(data_to_load1)
        for peptide in sequences:
            amino_acids_ap = split_amino_acids(peptide, amino_acids_AP)
            dipeptides_ap = split_dipeptides(peptide, dipeptides_AP)
            tripeptides_ap = split_tripeptides(peptide, tripeptides_AP)
            
            new_data_ansamble.append([amino_acids_ap, dipeptides_ap, tripeptides_ap])
    else:
        amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP(data_to_load1)
        amino_acids_logP, dipeptides_logP, tripeptides_logP = load_data_AP(data_to_load2)
        for peptide in sequences:
            amino_acids_ap = split_amino_acids(peptide, amino_acids_AP)
            dipeptides_ap = split_dipeptides(peptide, dipeptides_AP)
            tripeptides_ap = split_tripeptides(peptide, tripeptides_AP)
             
            amino_acids_logp = split_amino_acids(peptide, amino_acids_logP)
            dipeptides_logp  = split_dipeptides(peptide, dipeptides_logP)
            tripeptides_logp  = split_tripeptides(peptide, tripeptides_logP)
            
            new_data_ansamble.append([[amino_acids_ap, amino_acids_logp], [dipeptides_ap, dipeptides_logp], [tripeptides_ap, tripeptides_logp]])

    # Generate predictions on data that has no label beforehand 
    model_predictions_amino = model_predict(new_data_ansamble, labels, best_model_file_amino, model_type=0, merge=merge)
    model_predictions_di = model_predict(new_data_ansamble, labels, best_model_file_di, model_type=1, merge=merge)
    model_predictions_tri = model_predict(new_data_ansamble, labels, best_model_file_tri, model_type=2, merge=merge)
    model_predictions_ansamble = model_predict(new_data_ansamble, labels, best_model_file_ansamble, model_type=-1, merge=merge)

    model_predictions_voting = average_model(model_predictions_amino, model_predictions_di, model_predictions_tri, use_binary=True)
    model_predictions_avg = average_model(model_predictions_amino, model_predictions_di, model_predictions_tri)

    # Write SA probability to file
    percentage_filename = ""
    if merge: 
        percentage_filename = MERGE_MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+"percentage_merged_" + data_to_load + ".csv"
    else: 
        percentage_filename = MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+"percentage_" + data_to_load + ".csv"
    percentage_file = open(percentage_filename, "w", encoding="utf-8")
    percentage_string_to_write = "Sekvenca;Model aminokiselina;Model dipeptida;Model tripeptida;Model ansambla;Model glasovanja;Model prosjeka;Metoda bez RNN-a\n"
    for x in range(len(labels)):
        percentage_string_to_write += sequences[x]+";"+str(np.round(model_predictions_amino[x] * 100, 2))+";"+str(np.round(model_predictions_di[x] * 100, 2))+";"+str(round(model_predictions_tri[x] * 100, 2))+";"+str(np.round(model_predictions_ansamble[x] * 100, 2))+";"+str(np.round(model_predictions_voting[x] * 100, 2))+";"+str(np.round(model_predictions_avg[x] * 100, 2))+";"+past_grades[x]+"\n"
    percentage_string_to_write = percentage_string_to_write.replace('.',',')
    percentage_file.write(percentage_string_to_write)
    percentage_file.close()

    # Write class based on the threshold of probability to file
    threshold_amino = 0.5
    threshold_di = 0.5
    threshold_tri = 0.5
    threshold_ansamble = 0.5
    threshold_voting = 0.5
    threshold_average = 0.5

    model_predictions_amino = convert_to_binary(model_predictions_amino, threshold_amino)
    model_predictions_di = convert_to_binary(model_predictions_di, threshold_di)
    model_predictions_tri = convert_to_binary(model_predictions_tri, threshold_tri)
    model_predictions_ansamble = convert_to_binary(model_predictions_ansamble, threshold_ansamble)
    model_predictions_voting = convert_to_binary(model_predictions_voting, threshold_voting)
    model_predictions_avg = convert_to_binary(model_predictions_avg, threshold_average)
    
    grade_filename = ""
    if merge:
        grade_filename = MERGE_MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+"grade_merged_" + data_to_load + ".csv"
    else:
        grade_filename = MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+"grade_" + data_to_load + ".csv"
    grade_file = open(grade_filename, "w", encoding="utf-8")
    grade_string_to_write = "Sekvenca;Model aminokiselina;Model dipeptida;Model tripeptida;Model ansambla;Model glasovanja;Model prosjeka;Metoda bez RNN-a;Broj metoda koje se poklapaju s metodom bez RNN-a\n"
    correct_amino = 0
    correct_di = 0
    correct_tri = 0
    correct_ansamble = 0
    correct_voting = 0
    correct_average = 0
    for x in range(len(labels)):
        correct_in_row = 0
        if (model_predictions_amino[x] == 1 and past_classes[x] == 'Y') or (model_predictions_amino[x] == 0 and past_classes[x] == 'N'):
            correct_amino += 1
            correct_in_row += 1
        if (model_predictions_di[x] == 1 and past_classes[x] == 'Y') or (model_predictions_di[x] == 0 and past_classes[x] == 'N'):
            correct_di += 1
            correct_in_row += 1
        if (model_predictions_tri[x] == 1 and past_classes[x] == 'Y') or (model_predictions_tri[x] == 0 and past_classes[x] == 'N'):
            correct_tri += 1
            correct_in_row += 1 
        if (model_predictions_ansamble[x] == 1 and past_classes[x] == 'Y') or (model_predictions_ansamble[x] == 0 and past_classes[x] == 'N'):
            correct_ansamble += 1
            correct_in_row += 1
        if (model_predictions_voting[x] == 1 and past_classes[x] == 'Y') or (model_predictions_voting[x] == 0 and past_classes[x] == 'N'):
            correct_voting += 1
            correct_in_row += 1
        if (model_predictions_voting[x] == 1 and past_classes[x] == 'Y') or (model_predictions_voting[x] == 0 and past_classes[x] == 'N'):
            correct_average += 1
            correct_in_row += 1
        part1 = sequences[x]+";"+str(model_predictions_amino[x])+";"+str(model_predictions_di[x])+";"+str(model_predictions_tri[x])+";"+str(model_predictions_ansamble[x])+";"+str(model_predictions_voting[x])+";"+str(model_predictions_avg[x])+";"+past_classes[x]
        part1 = part1.replace(".0",'')
        part1 = part1.replace('1','Y')
        part1 = part1.replace('0','N')
        part2 = ";"+str(correct_in_row)+"\n"
        part2 = part2.replace(".",',')
        grade_string_to_write += part1 + part2 
    last_line = "Broj poklapanja s metodom bez RNN-a po metodi"+";"+str(correct_amino)+";"+str(correct_di)+";"+str(correct_tri)+";"+str(correct_ansamble)+";"+str(correct_voting)+";"+str(correct_average)+";"+""+";"+""+"\n"
    last_line = last_line.replace(".",',') 
    grade_string_to_write += last_line
    grade_file.write(grade_string_to_write)
    grade_file.close()
    
def after_training_no_model_type(test_number, best_model_file_amino, model_type=-1, data_to_load="ALL_PROPERTIES", merge = False, no_model_type = False):
        
        # Get sequences for peptides with no labels and predictions from the model without machine learning 
        resulteval = DATA_PATH+'RESULTEVAL.csv' 
        df = pd.read_csv(resulteval, skipinitialspace=True, sep=';')
        sequences = list(df['Dizajnirani peptid'])
        past_grades = list(df['Postotak svojstava koja imaju AUC > 0,5 koja su dala SA'])
        past_classes= list(df['SA'])
      
        labels = np.zeros(len(sequences))
        new_data_ansamble = []
        # Encode sequences
        encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-1, 1))) 
        if model_type == 0:
            for seq in sequences: 
                encoded = encoder.encode([seq])[0]  
                amino_removed = [code[1:39] for code in encoded] 
                new_data_ansamble.append(amino_removed)    
        else:
            amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP("AP") 
            for peptide in sequences:
                amino_acids_ap = split_amino_acids(peptide, amino_acids_AP)
                dipeptides_ap = split_dipeptides(peptide, dipeptides_AP)
                tripeptides_ap = split_tripeptides(peptide, tripeptides_AP)
                encoded = encoder.encode([peptide])[0] 
                amino_removed = [code[1:39] for code in encoded]  
                new_data_ansamble.append([amino_acids_ap, dipeptides_ap, tripeptides_ap, amino_removed]) 
      
        # Generate predictions on data that has no label beforehand 
        model_predictions_amino = model_predict(new_data_ansamble, labels, best_model_file_amino, model_type=model_type, no_model_type=no_model_type, merge=merge)
        # Write SA probability to file
        
        percentage_filename = MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+"percentage_" + data_to_load + ".csv"
        percentage_file = open(percentage_filename, "w", encoding="utf-8")
        percentage_string_to_write = "Sekvenca;Model svih svojstava;Metoda bez RNN-a\n"
        if data_to_load != "ALL_PROPERTIES":
            percentage_string_to_write = "Sekvenca;Model svih svojstava ansambl;Metoda bez RNN-a\n"
        for x in range(len(labels)):
            percentage_string_to_write += sequences[x]+";"+str(np.round(model_predictions_amino[x] * 100, 2))+";"+past_grades[x]+"\n"
        percentage_string_to_write = percentage_string_to_write.replace('.',',')
        percentage_file.write(percentage_string_to_write)
        percentage_file.close()

        # Write class based on the threshold of probability to file
        
        threshold_amino = 0.5 
        
        model_predictions_amino = convert_to_binary(model_predictions_amino, threshold_amino) 
        
        grade_filename = MODEL_DATA_PATH[data_to_load]+str(test_number)+"_"+"grade_" + data_to_load + ".csv"
        grade_file = open(grade_filename, "w", encoding="utf-8")
        grade_string_to_write = "Sekvenca;Model svih svojstava;Metoda bez RNN-a\n"
        if data_to_load != "ALL_PROPERTIES":
            grade_string_to_write = "Sekvenca;Model svih svojstava ansambl;Metoda bez RNN-a\n"
        correct_amino = 0 
        for x in range(len(labels)): 
            if (model_predictions_amino[x] == 1 and past_classes[x] == 'Y') or (model_predictions_amino[x] == 0 and past_classes[x] == 'N'):
                correct_amino += 1 
                
            part1 = sequences[x]+";"+str(model_predictions_amino[x])+";"+past_classes[x]+"\n"
            part1 = part1.replace(".0",'')
            part1 = part1.replace('1','Y')
            part1 = part1.replace('0','N') 
            grade_string_to_write += part1  
        last_line = "Broj poklapanja s metodom bez RNN-a;"+str(correct_amino)+";\n"
        last_line = last_line.replace(".",',') 
        grade_string_to_write += last_line
        grade_file.write(grade_string_to_write)
        grade_file.close() 