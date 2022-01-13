# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 22:08:47 2022

@author: USER07
"""

from keras.datasets import mnist

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

train_images.shape

len(train_images.shape)

test_images.shape

len(test_images)


#Network Architecture

from keras import models
from keras import layers

network=models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer='rmsprop',loss="categorical_crossentropy",metrics=['accuracy'])

train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255

test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255

from tensorflow.keras.utils import to_categorical 

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

network.fit(train_images,train_labels,epochs=5,batch_size=128)

test_loss, test_acc=network.evaluate(test_images,test_labels)


from keras.datasets import mnist

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

train_images.ndim
train_images.dtype

digit=train_images[1100]

import matplotlib.pyplot as plt

plt.imshow(digit,cmap=plt.cm.binary)

plt.show()

import numpy as np

#test

def test(x):
    
    assert x>10
    
    return np.sqrt(x)



def test1(a):
    a+=1
    return a
    

x=5

m=np.random.randint(1,300,size=120)
m1=m.reshape((30,4))
m2=m.reshape((15,2,4))
m3=m.reshape((5,3,2,4))
m4=m.reshape((5,4,3,2))


n=np.random.randint(1,300,size=(2,3,3))
n1=n.reshape(2,3*3)
































