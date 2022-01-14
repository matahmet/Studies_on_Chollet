# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 00:26:33 2022

@author: USER07
"""

from keras.datasets import imdb


df=imdb.load_data()
df[0][0][0]

word_index=imdb.get_word_index()
word_index["get"]
word_index["the"]
word_index["bad"]
word_index["good"]
word_index["crazy"]
word_index["turkish"]
word_index["york"]
word_index["california"]
word_index["francisco"]
word_index["arab"]
word_index["muslim"]
word_index["islam"]
word_index["of"]
word_index["ken"]

word_index["should"]
word_index["br"]

len(df[0][0][0])
len(set(df[0][0][0]))

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

len(train_data[0])
len(set(train_data[0]))

reverse_word_index=dict([(value,key) for (key,value) in word_index.items() ]  )
reverse_word_index[1]
reverse_word_index[2]
reverse_word_index[3]
reverse_word_index[4]
reverse_word_index[5]
reverse_word_index[6]

reverse_word_index[9999]
reverse_word_index[9998]
reverse_word_index[9997]
reverse_word_index[10000]
reverse_word_index[10001]

train_data[0]

max(train_data[0])


decoded_review=' '.join([reverse_word_index.get(i-3,'?') for i in train_data[20]])
decoded_review

decoded_review=' '.join([reverse_word_index.get(i-3,'?') for i in train_data[230]])
decoded_review

import numpy as np

def vectorize_sequence(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence]=1
    
    return results
    
            
  
            
            
            
            
            
            
            
            
            
            
    













