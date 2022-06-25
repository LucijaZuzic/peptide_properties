import keras
import tensorflow as tf
import numpy as np
from custom_plots import plt_model, plt_model_final, decorate_stats, decorate_stats_avg, decorate_stats_final
from utils import DATA_PATH, MODEL_DATA_PATH, SEQ_MODEL_DATA_PATH, MY_MODEL_DATA_PATH, scale, split_amino_acids, split_dipeptides, split_tripeptides, padding
import new_model
from sklearn.preprocessing import MinMaxScaler 
from seqprops import SequentialPropertiesEncoder 

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

def model_training_seq(test_number, train_and_validation_data, train_and_validation_labels, kfold_second, epochs, factor_NSA, mask_value=2):
    
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
                                # This callback will stop the training when there is no improvement in
                                # the loss for three consecutive epochs.
                                # Restoring best weights in case of performance drop
                                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
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
                    
    train_and_validation_data, train_and_validation_labels = reshape_seq(train_and_validation_data, train_and_validation_labels)
                
    model_picture = SEQ_MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_final_model.png'
    model_file = SEQ_MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_final_model.h5'
    
    #  Choose correct model and instantiate model
    
    model = new_model.create_seq_model(input_shape=np.shape(train_data[0]), conv1_filters=best_conv, conv2_filters=best_conv, conv_kernel_size=best_kernel, num_cells=best_numcells, dropout=best_dropout, mask_value=mask_value)

    # Save graphical representation of the model to a file.
    tf.keras.utils.plot_model(model, to_file=model_picture, show_shapes=True)
    
    # Print model summary.
    model.summary()
    
    callbacks = [
        # When validation loss doesn't decrease in 10 consecutive epochs, reduce the learning rate by 90%.
        # This is repeated while learning rate is >= 0.000001.
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=10, min_lr=0.000001
        ),
        # Save the best model (the one with the lowest validation loss).
        keras.callbacks.ModelCheckpoint(
            model_file, save_best_only=True, monitor='loss', mode='min'
        ),
        # This callback will stop the training when there is no improvement in
        # the loss for three consecutive epochs.
        # Restoring best weights in case of performance drop
        keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
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
        train_and_validation_data, 
        train_and_validation_labels,
        #validation_split = 0.1, 
        class_weight={0: factor_NSA, 1: 1.0},
        epochs=epochs,
        batch_size = best_batch_size,
        callbacks=callbacks,
        verbose=1
    ) 
                    
    print("Test %d best params: conv: %d num_cells: %d kernel_size: %d batch_size: %d dropout: %.2f" % (test_number, best_conv, best_numcells, best_kernel, best_batch_size, best_dropout))
    
    # Output accuracy, validation accuracy, loss and validation loss for all models
    decorate_stats_final(history) 

    # Plot the history
    plt_model_final(SEQ_MODEL_DATA_PATH, test_number, history, 'rnn_model_'+model_name+'_final_model') 
    
    return best_batch_size
    
    
def model_training(num_props, test_number, train_and_validation_data, train_and_validation_labels, kfold_second, epochs, factor_NSA, mask_value=2):
    
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
                                            # This callback will stop the training when there is no improvement in
                                            # the loss for three consecutive epochs.
                                            # Restoring best weights in case of performance drop
                                            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
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
                        
    train_and_validation_data, train_and_validation_labels = reshape(num_props, train_and_validation_data, train_and_validation_labels)
                    
    model_picture = MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_final_model.png'
    model_file = MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_final_model.h5'
    
    #  Choose correct model and instantiate model
    model = new_model.amino_di_tri_model(num_props, input_shape = np.shape(train_data[num_props][0]), conv=best_conv, numcells=best_numcells, kernel_size = best_kernel,  lstm1=best_lstm, lstm2=best_lstm, dense=best_dense, dropout=best_dropout, lambda2=best_lambda, mask_value=mask_value)

 
    # Save graphical representation of the model to a file.
    tf.keras.utils.plot_model(model, to_file=model_picture, show_shapes=True)
    
    # Print model summary.
    model.summary()
    
    callbacks = [
        # When validation loss doesn't decrease in 10 consecutive epochs, reduce the learning rate by 90%.
        # This is repeated while learning rate is >= 0.000001.
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=10, min_lr=0.000001
        ),
        # Save the best model (the one with the lowest validation loss).
        keras.callbacks.ModelCheckpoint(
            model_file, save_best_only=True, monitor='loss', mode='min'
        ),
        # This callback will stop the training when there is no improvement in
        # the loss for three consecutive epochs.
        # Restoring best weights in case of performance drop
        keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
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
        train_and_validation_data, 
        train_and_validation_labels,
        #validation_split = 0.1, 
        class_weight={0: factor_NSA, 1: 1.0},
        epochs=epochs,
        batch_size = best_batch_size,
        callbacks=callbacks,
        verbose=1
    ) 
                    
    print("Test %d best params: conv: %d num_cells: %d kernel_size: %d lstm: %d dense: %d lambda: %.2f dropout: %.2f batch: %d" % (test_number, best_conv, best_numcells, best_kernel, best_lstm, best_dense, best_lambda, best_dropout, best_batch_size))
    
    # Output accuracy, validation accuracy, loss and validation loss for all models
    decorate_stats_final(history) 

    # Plot the history
    plt_model_final(MODEL_DATA_PATH, test_number, history, 'rnn_model_'+model_name+'_final_model') 
    
    return best_batch_size

def model_training_AP(num_props, test_number, train_and_validation_data, train_and_validation_labels, kfold_second, epochs, factor_NSA, mask_value=2):
    
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
                                # This callback will stop the training when there is no improvement in
                                # the loss for three consecutive epochs.
                                # Restoring best weights in case of performance drop
                                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
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
                        
    train_and_validation_data, train_and_validation_labels = reshape_AP(num_props, train_and_validation_data, train_and_validation_labels)
                    
    model_picture = MY_MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_final_model.png'
    model_file = MY_MODEL_DATA_PATH+str(test_number)+'_rnn_model_'+model_name+'_final_model.h5'
    
    #  Choose correct model and instantiate model
    model = new_model.only_amino_di_tri_model(num_props, lstm1=best_lstm, lstm2=best_lstm, dense=best_dense, dropout=best_dropout, lambda2=best_lambda, mask_value=mask_value)

 
    # Save graphical representation of the model to a file.
    tf.keras.utils.plot_model(model, to_file=model_picture, show_shapes=True)
    
    # Print model summary.
    model.summary()
    
    callbacks = [
        # When validation loss doesn't decrease in 10 consecutive epochs, reduce the learning rate by 90%.
        # This is repeated while learning rate is >= 0.000001.
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=10, min_lr=0.000001
        ),
        # Save the best model (the one with the lowest validation loss).
        keras.callbacks.ModelCheckpoint(
            model_file, save_best_only=True, monitor='loss', mode='min'
        ),
        # This callback will stop the training when there is no improvement in
        # the loss for three consecutive epochs.
        # Restoring best weights in case of performance drop
        keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
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
        train_and_validation_data, 
        train_and_validation_labels,
        #validation_split = 0.1, 
        class_weight={0: factor_NSA, 1: 1.0},
        epochs=epochs,
        batch_size = best_batch_size,
        callbacks=callbacks,
        verbose=1
    ) 
                    
    print("Test %d best params: lstm: %d dense: %d lambda: %.2f dropout: %.2f batch: %d" % (test_number, best_lstm, best_dense, best_lambda, best_dropout, best_batch_size))
    
    # Output accuracy, validation accuracy, loss and validation loss for all models
    decorate_stats_final(history) 

    # Plot the history
    plt_model_final(MY_MODEL_DATA_PATH, test_number, history, 'rnn_model_'+model_name+'_final_model') 
    
    return best_batch_size