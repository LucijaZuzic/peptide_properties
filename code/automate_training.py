from custom_plots import convert_to_binary
import keras
import tensorflow as tf
import numpy as np
from custom_plots import plt_model
from data_generator import DataGenerator
from utils import DATA_PATH, MODEL_DATA_PATH, MERGE_MODEL_DATA_PATH, scale, split_amino_acids, split_dipeptides, split_tripeptides
import models
from sklearn.preprocessing import MinMaxScaler 
from seqprops import SequentialPropertiesEncoder 

def data_and_labels_from_indices(all_data, all_labels, indices, data_index = -1):
    data = []
    labels = []

    for i in indices:
        if data_index != -1:
            data.append(all_data[i][data_index])
        else:
            data.append(all_data[i])
        labels.append(all_labels[i]) 
    
    return data, labels 

# Choose loading AP, APH, logP or polarity
def load_data_AP(data_to_load="AP"):
    # Load AP scores. 
    amino_acids_AP = np.load(DATA_PATH+'amino_acids_' + data_to_load + '.npy', allow_pickle=True).item()
    dipeptides_AP = np.load(DATA_PATH+'dipeptides_' + data_to_load + '.npy', allow_pickle=True).item()
    tripeptides_AP = np.load(DATA_PATH+'tripeptides_' + data_to_load + '.npy', allow_pickle=True).item()
    
    # Scale scores to range [-0.5, 0.5].
    scale(amino_acids_AP)
    scale(dipeptides_AP)
    scale(tripeptides_AP)

    return amino_acids_AP, dipeptides_AP, tripeptides_AP

# Choose loading AP, APH, logP or polarity
def load_data_SA(data_to_load="AP"):
    amino_acids_AP, dipeptides_AP, tripeptides_AP = [], [], []
    if data_to_load == "AP_ALL_PROPERTIES":
        amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP("AP")
    else:
        amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP(data_to_load)

    # Load self-assembly data.
    SA_data = np.load(DATA_PATH+'data_SA.npy')

    # Split peptides in two bins.
    # SA - has self-assembly, NSA - does not have self-assembly.
    SA = []
    NSA = []
    
    encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-1, 1)))
     
    # Convert each peptide sequence to three lists of AP scores: amino acid, dipeptide and tripeptide scores.
    for peptide in SA_data:
        amino_acids_ap = split_amino_acids(peptide[0], amino_acids_AP)
        dipeptides_ap = split_dipeptides(peptide[0], dipeptides_AP)
        tripeptides_ap = split_tripeptides(peptide[0], tripeptides_AP)
        
        if peptide[1] == '1': 
            if data_to_load == "AP_ALL_PROPERTIES": 
                encoded = encoder.encode([peptide[0]])[0] 
                amino_removed = [code[1:39] for code in encoded]  
                SA.append([amino_acids_ap, dipeptides_ap, tripeptides_ap, amino_removed]) 
            else:
                SA.append([amino_acids_ap, dipeptides_ap, tripeptides_ap])
        elif peptide[1] == '0':
            if data_to_load == "AP_ALL_PROPERTIES": 
                encoded = encoder.encode([peptide[0]])[0] 
                amino_removed = [code[1:39] for code in encoded]  
                NSA.append([amino_acids_ap, dipeptides_ap, tripeptides_ap, amino_removed]) 
            else:
                NSA.append([amino_acids_ap, dipeptides_ap, tripeptides_ap]) 
                
    return SA, NSA
 
# Choose loading AP, APH, logP or polarity
def load_data_SA_merged(data1, data2):
    amino_acids_Data1, dipeptides_Data1, tripeptides_Data1 = load_data_AP(data1)
    amino_acids_Data2, dipeptides_Data2, tripeptides_Data2 = load_data_AP(data2)

    # Load self-assembly data.
    SA_data = np.load(DATA_PATH+'data_SA.npy')

    # Split peptides in two bins.
    # SA - has self-assembly, NSA - does not have self-assembly.
    SA = []
    NSA = []
    # Convert each peptide sequence to three lists of AP scores: amino acid, dipeptide and tripeptide scores.
    for peptide in SA_data:
        amino_acids_data1 = split_amino_acids(peptide[0], amino_acids_Data1)
        dipeptides_data1 = split_dipeptides(peptide[0], dipeptides_Data1)
        tripeptides_data1 = split_tripeptides(peptide[0], tripeptides_Data1)
         
        amino_acids_data2 = split_amino_acids(peptide[0], amino_acids_Data2)
        dipeptides_data2 = split_dipeptides(peptide[0], dipeptides_Data2)
        tripeptides_data2 = split_tripeptides(peptide[0], tripeptides_Data2)
        
        if peptide[1] == '1':
            SA.append([[amino_acids_data1, amino_acids_data2], [dipeptides_data1, dipeptides_data2], [tripeptides_data1, tripeptides_data2]])
        elif peptide[1] == '0':
            NSA.append([[amino_acids_data1, amino_acids_data2], [dipeptides_data1, dipeptides_data2], [tripeptides_data1, tripeptides_data2]])
   
    return SA, NSA

def merge_data(SA, NSA):
    # Merge the bins and add labels
    merged_data = SA + NSA
    merged_labels = np.ones(len(SA) + len(NSA))
    merged_labels[len(SA):] *= 0

    return merged_data, merged_labels

def model_training(test_number, train_and_validation_data, train_and_validation_labels, kfold_second, epochs, factor_NSA, model_type=-1, data_to_load="AP", merge = False, function="selu", no_model_type= False):
    
    model_name = ""
    
    if (model_type == 0):
        model_name += 'amino'
    elif (model_type == 1):
        model_name += 'di'
    elif (model_type == 2):
        model_name += 'tri'
    else:
        model_name += 'ansamble'
    
    if (not merge):
        model_name += "_" + data_to_load
    else:
        model_name += "_merged_" + data_to_load
           
    fold_nr = 0
    best_model_index = 0
    min_val_loss = 1000
    model_history = []

    for train_data_indices, validation_data_indices in kfold_second.split(train_and_validation_data, train_and_validation_labels):
        # Save model to correct file based on number of fold
        fold_nr += 1
        
        if not merge:
            model_picture = MODEL_DATA_PATH[data_to_load]+str(test_number)+'_rnn_model_'+model_name+'_'+str(fold_nr)+'.png'
            model_file = MODEL_DATA_PATH[data_to_load]+str(test_number)+'_best_model_'+model_name+'_'+str(fold_nr)+'.h5'
        else:
            model_picture = MERGE_MODEL_DATA_PATH[data_to_load]+str(test_number)+'_rnn_model_'+model_name+'_'+str(fold_nr)+'.png'
            model_file = MERGE_MODEL_DATA_PATH[data_to_load]+str(test_number)+'_best_model_'+model_name+'_'+str(fold_nr)+'.h5'
            
        # Convert train indices to train data and train labels
        train_data, train_labels = data_and_labels_from_indices(train_and_validation_data, train_and_validation_labels, train_data_indices, data_index = -1)
        
        # Convert validation indices to validation data and validation labels
        val_data, val_labels = data_and_labels_from_indices(train_and_validation_data, train_and_validation_labels, validation_data_indices, data_index = -1)
        
        #  Choose correct model and instantiate model
        if not merge:
            if function == "selu":
                if (model_type == 0):
                    model = models.amino_model()
                elif (model_type == 1):
                    model = models.dipeptide_model()
                elif (model_type == 2):
                    model = models.tripeptide_model()
                else:
                    model = models.amino_di_tri_model()
            elif function == "relu":
                if (model_type == 0):
                    model = models.amino_model_polarity()
                elif (model_type == 1):
                    model = models.dipeptide_model_polarity()
                elif (model_type == 2):
                    model = models.tripeptide_model_polarity()
                else:
                    model = models.amino_di_tri_model_polarity()
        else:
            if function == "selu":
                if (model_type == 0):
                    model = models.amino_merge_model()
                elif (model_type == 1):
                    model = models.dipeptide_merge_model()
                elif (model_type == 2):
                    model = models.tripeptide_merge_model()
                else:
                    model = models.amino_di_tri_merge_model()
            elif function == "relu":
                if (model_type == 0):
                    model = models.amino_merge_polarity_model()
                elif (model_type == 1):
                    model = models.dipeptide_merge_polarity_model()
                elif (model_type == 2):
                    model = models.tripeptide_merge_polarity_model()
                else:
                    model = models.amino_di_tri_merge_polarity_model()
        if no_model_type:
            if model_type == 0:
                model = models.multiple_property_model() 
            else:
                model = models.multiple_property_model_concat() 

        # Save graphical representation of the model to a file.
        tf.keras.utils.plot_model(model, to_file=model_picture, show_shapes=True)
        
        # Print model summary.
        model.summary()
        
        callbacks = [
            # When validation loss doesn't decrease in 10 consecutive epochs, reduce the learning rate by 90%.
            # This is repeated while learning rate is >= 0.000001.
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.1, patience=10, min_lr=0.000001
            ),
            # Save the best model (the one with the lowest validation loss).
            keras.callbacks.ModelCheckpoint(
                model_file, save_best_only=True, monitor='val_loss', mode='min'
            ),
        ]

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Check if the model is ansamble or single
        number_items = 1
        if model_type == -1:
            if merge:
                number_items = 6
            else:
                number_items = 3
        elif merge:
            number_items = 2
        if no_model_type:
            if model_type == 0:
                number_items = 0
            else:
                number_items = 4

        # Train the model.
        # After model training, the `history` variable will contain important parameters for each epoch, such as
        # train loss, train accuracy, learning rate, and so on.
        history = model.fit(
            DataGenerator(train_data, train_labels, number_items=number_items),
            validation_data=DataGenerator(val_data, val_labels, number_items=number_items),
            class_weight={0: factor_NSA, 1: 1.0},
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        model_history.append(history)

        # Plot the history
        plt_model(test_number, history, 'rnn_model_'+model_name+'_'+str(fold_nr), data_to_load=data_to_load, merge=merge)

        # Save the file name of the best fold (the one with the lowest validation loss).

        if (np.min(history.history['val_loss']) < min_val_loss):
            min_val_loss = np.min(history.history['val_loss'])
            best_model_index = fold_nr 
    
    return best_model_index, model_history

def average_model(model_predictions_amino, model_predictions_di, model_predictions_tri, 
                    use_binary=False, weight_amino=1.0/3.0, weight_di=1.0/3.0, weight_tri=1.0/3.0, 
                    threshold_amino=0.5, threshold_di=0.5, threshold_tri=0.5):

    if use_binary:
        model_predictions_amino = convert_to_binary(model_predictions_amino, threshold_amino)
        model_predictions_di = convert_to_binary(model_predictions_di, threshold_di)
        model_predictions_tri = convert_to_binary(model_predictions_tri, threshold_tri)

    model_predictions_avg = []

    model_predictions_avg = [(weight_amino * model_predictions_amino[i] + weight_di * model_predictions_di[i] + weight_tri * model_predictions_tri[i]) for i in range(len(model_predictions_amino))]
    
    return model_predictions_avg