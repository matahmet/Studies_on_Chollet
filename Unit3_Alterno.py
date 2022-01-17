# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 00:16:55 2022

@author: USER07
"""

from keras.datasets import imdb

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

import numpy as np

def vectorize_sequence(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence]=1
    
    return results

x_train=vectorize_sequence(train_data)       

x_test=vectorize_sequence(test_data)

y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')

from tensorflow import keras
from tensorflow.keras import layers

model=keras.Sequential([
    layers.Dense(16,activation="relu"),
    layers.Dense(16,activation="relu"),
    layers.Dense(1,activation='sigmoid')    
    ])

 