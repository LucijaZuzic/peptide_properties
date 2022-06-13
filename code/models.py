#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
import numpy as np
# LAMBDA controls the lambda factor in L2 regularization.
LAMBDA = 0.0
# DROPOUT_PERCENT controls the fraction of the input units to drop during drop.

DROPOUT_PERCENT_AMINO = 0.2
DROPOUT_PERCENT_DI = 0.2
DROPOUT_PERCENT_TRI = 0.2

AMINO_LSTM = 5
AMINO_DENSE = 15

DI_LSTM = 5
DI_DENSE = 15

TRI_LSTM = 5
TRI_DENSE = 15

def __multiple_property_model():
    # LSTM model which processes sequence props

    input_layer = keras.layers.Input((None, 1))
    lstm_layer_1 = keras.layers.Bidirectional(keras.layers.LSTM(AMINO_LSTM, return_sequences=True))(input_layer)
    lstm_layer_2 = keras.layers.LSTM(AMINO_LSTM)(lstm_layer_1)

    dense_layer1 = keras.layers.Dense(AMINO_DENSE, activation='selu', 
                                      kernel_regularizer=keras.regularizers.l2(l=LAMBDA))(lstm_layer_2)
    dropout_1 = keras.layers.Dropout(DROPOUT_PERCENT_AMINO)(dense_layer1)
  
    embedding_sum_one_axis = keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=0, keepdims=True))(dropout_1) 

    return input_layer, embedding_sum_one_axis

def multiple_property_model():
    multiple_property_layer_input, multiple_property_layer_output = __multiple_property_model()
    
    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(multiple_property_layer_output)
    	
    # The final model accepts a list of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=multiple_property_layer_input,
                              outputs=final_output_layer)

def multiple_property_model_concat():
    # Instantiate three separate submodels.
    amino_layer_input, amino_layer_output = __amino_model()
    dipeptide_layer_input, dipeptide_layer_output = __dipeptide_model()
    tripeptide_layer_input, tripeptide_layer_output = __tripeptide_model()
    multiple_property_layer_input, multiple_property_layer_output = __multiple_property_model()

    # Merge the submodels.
    merge_layer = keras.layers.Concatenate()(
        [amino_layer_output,
         dipeptide_layer_output,
         tripeptide_layer_output,
         multiple_property_layer_output]
    )

    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(merge_layer)

    # The final model accepts three lists of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=[amino_layer_input,
                                      dipeptide_layer_input,
                                      tripeptide_layer_input,
                                      multiple_property_layer_input],
                              outputs=final_output_layer)

def __amino_model_polarity():
    # LSTM model which processes amino acid AP scores.

    input_layer = keras.layers.Input((None, 1))
    lstm_layer_1 = keras.layers.Bidirectional(keras.layers.LSTM(AMINO_LSTM, return_sequences=True))(input_layer)
    lstm_layer_2 = keras.layers.LSTM(AMINO_LSTM)(lstm_layer_1)

    dense_layer1 = keras.layers.Dense(AMINO_DENSE, activation='relu', 
                                      kernel_regularizer=keras.regularizers.l2(l=LAMBDA))(lstm_layer_2)
    dropout_1 = keras.layers.Dropout(DROPOUT_PERCENT_AMINO)(dense_layer1)

    return input_layer, dropout_1

def __amino_model():
    # LSTM model which processes amino acid AP scores.

    input_layer = keras.layers.Input((None, 1))
    lstm_layer_1 = keras.layers.Bidirectional(keras.layers.LSTM(AMINO_LSTM, return_sequences=True))(input_layer)
    lstm_layer_2 = keras.layers.LSTM(AMINO_LSTM)(lstm_layer_1)

    dense_layer1 = keras.layers.Dense(AMINO_DENSE, activation='selu', 
                                      kernel_regularizer=keras.regularizers.l2(l=LAMBDA))(lstm_layer_2)
    dropout_1 = keras.layers.Dropout(DROPOUT_PERCENT_AMINO)(dense_layer1)

    return input_layer, dropout_1

def __dipeptide_model_polarity():
    # LSTM model which processes dipeptide AP scores.

    input_layer = keras.layers.Input((None, 1))
    lstm_layer_1 = keras.layers.Bidirectional(keras.layers.LSTM(5, return_sequences=True))(input_layer)
    lstm_layer_2 = keras.layers.LSTM(DI_LSTM)(lstm_layer_1)

    dense_layer1 = keras.layers.Dense(DI_DENSE, activation='relu',
                                      kernel_regularizer=keras.regularizers.l2(l=LAMBDA))(lstm_layer_2)
    dropout_1 = keras.layers.Dropout(DROPOUT_PERCENT_DI)(dense_layer1)

    return input_layer, dropout_1

def __dipeptide_model():
    # LSTM model which processes dipeptide AP scores.

    input_layer = keras.layers.Input((None, 1))
    lstm_layer_1 = keras.layers.Bidirectional(keras.layers.LSTM(5, return_sequences=True))(input_layer)
    lstm_layer_2 = keras.layers.LSTM(DI_LSTM)(lstm_layer_1)

    dense_layer1 = keras.layers.Dense(DI_DENSE, activation='selu',
                                      kernel_regularizer=keras.regularizers.l2(l=LAMBDA))(lstm_layer_2)
    dropout_1 = keras.layers.Dropout(DROPOUT_PERCENT_DI)(dense_layer1)

    return input_layer, dropout_1

def __tripeptide_model_polarity():
    # LSTM model which processes tripeptide AP scores.

    input_layer = keras.layers.Input((None, 1))
    lstm_layer_1 = keras.layers.Bidirectional(keras.layers.LSTM(5, return_sequences=True))(input_layer)
    lstm_layer_2 = keras.layers.LSTM(TRI_LSTM)(lstm_layer_1)

    dense_layer1 = keras.layers.Dense(TRI_DENSE, activation='relu',
                                      kernel_regularizer=keras.regularizers.l2(l=LAMBDA))(lstm_layer_2)
    dropout_1 = keras.layers.Dropout(DROPOUT_PERCENT_TRI)(dense_layer1)

    return input_layer, dropout_1

def __tripeptide_model():
    # LSTM model which processes tripeptide AP scores.

    input_layer = keras.layers.Input((None, 1))
    lstm_layer_1 = keras.layers.Bidirectional(keras.layers.LSTM(5, return_sequences=True))(input_layer)
    lstm_layer_2 = keras.layers.LSTM(TRI_LSTM)(lstm_layer_1)

    dense_layer1 = keras.layers.Dense(TRI_DENSE, activation='selu',
                                      kernel_regularizer=keras.regularizers.l2(l=LAMBDA))(lstm_layer_2)
    dropout_1 = keras.layers.Dropout(DROPOUT_PERCENT_TRI)(dense_layer1)

    return input_layer, dropout_1

def amino_di_tri_model_polarity():
    # Instantiate three separate submodels.
    amino_layer_input, amino_layer_output = __amino_model_polarity()
    dipeptide_layer_input, dipeptide_layer_output = __dipeptide_model_polarity()
    tripeptide_layer_input, tripeptide_layer_output = __tripeptide_model_polarity()

    # Merge the submodels.
    merge_layer = keras.layers.Concatenate()(
        [amino_layer_output,
         dipeptide_layer_output,
         tripeptide_layer_output]
    )

    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(merge_layer)

    # The final model accepts three lists of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=[amino_layer_input,
                                      dipeptide_layer_input,
                                      tripeptide_layer_input],
                              outputs=final_output_layer)

def amino_di_tri_model():
    # Instantiate three separate submodels.
    amino_layer_input, amino_layer_output = __amino_model()
    dipeptide_layer_input, dipeptide_layer_output = __dipeptide_model()
    tripeptide_layer_input, tripeptide_layer_output = __tripeptide_model()

    # Merge the submodels.
    merge_layer = keras.layers.Concatenate()(
        [amino_layer_output,
         dipeptide_layer_output,
         tripeptide_layer_output]
    )

    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(merge_layer)

    # The final model accepts three lists of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=[amino_layer_input,
                                      dipeptide_layer_input,
                                      tripeptide_layer_input],
                              outputs=final_output_layer)

def amino_di_tri_merge_model():
    # Instantiate three separate submodels.
    amino_merge_layer_input, amino_merge_layer_output = __amino_merge_model()
    dipeptide_merge_layer_input, dipeptide_merge_layer_output = __dipeptide_merge_model()
    tripeptide_merge_layer_input, tripeptide_merge_layer_output = __tripeptide_merge_model()

    # Merge the submodels.
    merge_layer = keras.layers.Concatenate()(
        [amino_merge_layer_output,
         dipeptide_merge_layer_output,
         tripeptide_merge_layer_output]
    )

    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(merge_layer)

    # The final model accepts three lists of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=[amino_merge_layer_input,
                                      dipeptide_merge_layer_input,
                                      tripeptide_merge_layer_input],
                              outputs=final_output_layer)

def amino_di_tri_merge_polarity_model():
    # Instantiate three separate submodels.
    amino_merge_layer_input, amino_merge_layer_output = __amino_merge_polarity_model()
    dipeptide_merge_layer_input, dipeptide_merge_layer_output = __dipeptide_merge_polarity_model()
    tripeptide_merge_layer_input, tripeptide_merge_layer_output = __tripeptide_merge_polarity_model()

    # Merge the submodels.
    merge_layer = keras.layers.Concatenate()(
        [amino_merge_layer_output,
         dipeptide_merge_layer_output,
         tripeptide_merge_layer_output]
    )

    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(merge_layer)

    # The final model accepts three lists of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=[amino_merge_layer_input,
                                      dipeptide_merge_layer_input,
                                      tripeptide_merge_layer_input],
                              outputs=final_output_layer)

def __amino_merge_model():
    # Instantiate two separate submodels.
    amino_first_layer_input, amino_first_layer_output = __amino_model() 
    amino_second_layer_input, amino_second_layer_output = __amino_model() 

    # Merge the submodels.
    merge_layer = keras.layers.Concatenate()(
        [amino_first_layer_output,
         amino_second_layer_output]
    )
    
    return [amino_first_layer_input, amino_second_layer_input], merge_layer
   
def amino_merge_model():
    amino_merge_layer_input, amino_merge_layer_output = __amino_merge_model()
 
    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(amino_merge_layer_output)

    # The final model accepts three lists of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=amino_merge_layer_input,
                              outputs=final_output_layer)

def __amino_merge_polarity_model():
    # Instantiate two separate submodels.
    amino_first_layer_input, amino_first_layer_output = __amino_model() 
    amino_second_layer_input, amino_second_layer_output = __amino_model_polarity() 

    # Merge the submodels.
    merge_layer = keras.layers.Concatenate()(
        [amino_first_layer_output,
         amino_second_layer_output]
    )
    
    return [amino_first_layer_input, amino_second_layer_input], merge_layer
   
def amino_merge_polarity_model():
    amino_merge_layer_input, amino_merge_layer_output = __amino_merge_polarity_model()
 
    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(amino_merge_layer_output)

    # The final model accepts three lists of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=amino_merge_layer_input,
                              outputs=final_output_layer)

def __dipeptide_merge_model():
    # Instantiate two separate submodels.
    dipeptide_first_layer_input, dipeptide_first_layer_output = __dipeptide_model() 
    dipeptide_second_layer_input, dipeptide_second_layer_output = __dipeptide_model() 

    # Merge the submodels.
    merge_layer = keras.layers.Concatenate()(
        [dipeptide_first_layer_output,
         dipeptide_second_layer_output]
    )
    
    return [dipeptide_first_layer_input, dipeptide_second_layer_input], merge_layer
   
def dipeptide_merge_model():
    dipeptide_merge_layer_input, dipeptide_merge_layer_output = __dipeptide_merge_model()
 
    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(dipeptide_merge_layer_output)

    # The final model accepts three lists of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=dipeptide_merge_layer_input,
                              outputs=final_output_layer)

def __dipeptide_merge_polarity_model():
    # Instantiate two separate submodels.
    dipeptide_first_layer_input, dipeptide_first_layer_output = __dipeptide_model() 
    dipeptide_second_layer_input, dipeptide_second_layer_output = __dipeptide_model_polarity() 

    # Merge the submodels.
    merge_layer = keras.layers.Concatenate()(
        [dipeptide_first_layer_output,
         dipeptide_second_layer_output]
    )
    
    return [dipeptide_first_layer_input, dipeptide_second_layer_input], merge_layer
   
def dipeptide_merge_polarity_model():
    dipeptide_merge_layer_input, dipeptide_merge_layer_output = __dipeptide_merge_polarity_model()
 
    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(dipeptide_merge_layer_output)

    # The final model accepts three lists of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=dipeptide_merge_layer_input,
                              outputs=final_output_layer)

def __tripeptide_merge_model():
    # Instantiate two separate submodels.
    tripeptide_first_layer_input, tripeptide_first_layer_output = __tripeptide_model() 
    tripeptide_second_layer_input, tripeptide_second_layer_output = __tripeptide_model() 

    # Merge the submodels.
    merge_layer = keras.layers.Concatenate()(
        [tripeptide_first_layer_output,
         tripeptide_second_layer_output]
    )
    
    return [tripeptide_first_layer_input, tripeptide_second_layer_input], merge_layer
   
def tripeptide_merge_model():
    tripeptide_merge_layer_input, tripeptide_merge_layer_output = __tripeptide_merge_model()
 
    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(tripeptide_merge_layer_output)

    # The final model accepts three lists of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=tripeptide_merge_layer_input,
                              outputs=final_output_layer)

def __tripeptide_merge_polarity_model():
    # Instantiate two separate submodels.
    tripeptide_first_layer_input, tripeptide_first_layer_output = __tripeptide_model() 
    tripeptide_second_layer_input, tripeptide_second_layer_output = __tripeptide_model_polarity() 

    # Merge the submodels.
    merge_layer = keras.layers.Concatenate()(
        [tripeptide_first_layer_output,
         tripeptide_second_layer_output]
    )
    
    return [tripeptide_first_layer_input, tripeptide_second_layer_input], merge_layer
   
def tripeptide_merge_polarity_model():
    tripeptide_merge_layer_input, tripeptide_merge_layer_output = __tripeptide_merge_polarity_model()
 
    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(tripeptide_merge_layer_output)

    # The final model accepts three lists of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=tripeptide_merge_layer_input,
                              outputs=final_output_layer)

def amino_model_polarity():
    amino_layer_input, amino_layer_output = __amino_model_polarity()

    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(amino_layer_output)
    	
    # The final model accepts a list of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=amino_layer_input,
                              outputs=final_output_layer)

def amino_model():
    amino_layer_input, amino_layer_output = __amino_model()

    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(amino_layer_output)
    	
    # The final model accepts a list of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=amino_layer_input,
                              outputs=final_output_layer)

def dipeptide_model_polarity():
    dipeptide_layer_input, dipeptide_layer_output = __dipeptide_model_polarity()

    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(dipeptide_layer_output)

    # The final model accepts a list of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=dipeptide_layer_input,
                              outputs=final_output_layer)

def dipeptide_model():
    dipeptide_layer_input, dipeptide_layer_output = __dipeptide_model()

    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(dipeptide_layer_output)

    # The final model accepts a list of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=dipeptide_layer_input,
                              outputs=final_output_layer)

def tripeptide_model_polarity():
    tripeptide_layer_input, tripeptide_layer_output = __tripeptide_model_polarity()

    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(tripeptide_layer_output)

    # The final model accepts a list of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=tripeptide_layer_input,
                              outputs=final_output_layer)

def tripeptide_model():
    tripeptide_layer_input, tripeptide_layer_output = __tripeptide_model()

    final_output_layer = keras.layers.Dense(1, activation='sigmoid')(tripeptide_layer_output)

    # The final model accepts a list of AP scores as input, and returns a number in range [0, 1] which
    # indicates how probable it is the input sequence has self assembly.
    return keras.models.Model(inputs=tripeptide_layer_input,
                              outputs=final_output_layer)