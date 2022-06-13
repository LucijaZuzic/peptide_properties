#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    # A generator is used fetch the data for the model. Since the model's architecture is complex, we can not feed the
    # data in a conventional way, as a single numpy array, but we have to fetch the data on the go.

    def __init__(self, data, labels, number_items=True, shuffle=True):
        self.data = data
        self.labels = labels
        self.number_items = number_items

        # If True, indices are shuffled at the start of the training process and at the end of each epoch.
        self.shuffle = shuffle
        
        self.indices = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.data)

    def __getitem__(self, index):
        # Fetch a single item.

        # We use `self.indices` as a lookup table. If `self.shuffle` is True, `self.indices` will be shuffled
        # after each epoch so the data will be fetched in random order in each epoch.
        item_index = self.indices[index]

        return_data = []

        if self.number_items == 0: 
            return_data = np.array(self.data[item_index]).reshape(38, len(self.data[item_index]))
        elif self.number_items == 1:
            return_data = np.array(self.data[item_index]).reshape(1, len(self.data[item_index]))
        elif self.number_items == 2:
            AP_data = np.array(self.data[item_index][0]).reshape(1, len(self.data[item_index][0]))
            logP_data = np.array(self.data[item_index][1]).reshape(1, len(self.data[item_index][1]))
            return_data = [AP_data, logP_data]
        elif self.number_items == 3:
            amino_data = np.array(self.data[item_index][0]).reshape(1, len(self.data[item_index][0]))
            dipeptide_data = np.array(self.data[item_index][1]).reshape(1, len(self.data[item_index][1]))
            tripeptide_data = np.array(self.data[item_index][2]).reshape(1, len(self.data[item_index][2]))
            return_data = [amino_data, dipeptide_data, tripeptide_data]
        elif self.number_items == 4: 
            amino_data = np.array(self.data[item_index][0]).reshape(1, len(self.data[item_index][0]))
            dipeptide_data = np.array(self.data[item_index][1]).reshape(1, len(self.data[item_index][1]))
            tripeptide_data = np.array(self.data[item_index][2]).reshape(1, len(self.data[item_index][2]))
            all_data = np.array(self.data[item_index][3]).reshape(38, -1)
            return_data = [amino_data, dipeptide_data, tripeptide_data, all_data]
        elif self.number_items == 6:
            amino_data_data1 = np.array(self.data[item_index][0][0]).reshape(1, len(self.data[item_index][0][0]))
            amino_data_data2 = np.array(self.data[item_index][0][1]).reshape(1, len(self.data[item_index][0][1]))
            dipeptide_data_data1 = np.array(self.data[item_index][1][0]).reshape(1, len(self.data[item_index][1][0]))
            dipeptide_data_data2 = np.array(self.data[item_index][1][1]).reshape(1, len(self.data[item_index][1][1]))
            tripeptide_data_data1 = np.array(self.data[item_index][2][0]).reshape(1, len(self.data[item_index][2][0]))
            tripeptide_data_data2 = np.array(self.data[item_index][2][1]).reshape(1, len(self.data[item_index][2][1]))
            return_data = [amino_data_data1, amino_data_data2, dipeptide_data_data1, dipeptide_data_data2, tripeptide_data_data1, tripeptide_data_data2]

        label = np.array(self.labels[item_index]).reshape(1, 1) 
        return return_data, label

    def on_epoch_end(self):
        # Shuffle the indices.
        if self.shuffle:
            np.random.shuffle(self.indices)