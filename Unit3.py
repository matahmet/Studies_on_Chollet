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

len(df[0][0][0])
len(set(df[0][0][0]))

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

len(train_data[0])
len(set(train_data[0]))
