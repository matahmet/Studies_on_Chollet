# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 09:20:40 2022

@author: emrehan
"""
#get data
from keras.datasets import boston_housing

(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data()

#Normalizing the data
mean=train_data.mean(axis=0)
train_data-=mean

std=train_data.std(axis=0)
train_data/=std

test_data-=mean
test_data/=std

#model definition
from tensorflow import keras
from tensorflow.keras import layers

def build_model():
    model=keras.Sequential([
        layers.Dense(64,activation="relu"),
        layers.Dense(64,activation="relu"),
        layers.Dense(1)
        ])
    model.compile(optimizer="rmsprop",loss="mse",metrics=["mae"])
    return model

import numpy as np
k=4
num_val_samples=len(train_data)//k
num_epochs=100
all_scores=[]


for i in range(k):
    print(f"Processing fold #{i}")
    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets=train_targets[i*num_val_samples:(i+1)*num_val_samples]
    
    partial_train_data=np.concatenate(
        [train_data[:i*num_val_samples],
         train_data[(i+1)*num_val_samples:]],
        axis=0)
    
    partial_train_targets=np.concatenate(
        [train_targets[:i*num_val_samples],
         train_targets[(i+1)*num_val_samples:]],
        axis=0)
    
    
    model=build_model()
    model.fit(partial_train_data,partial_train_targets,
              epochs=num_epochs,batch_size=16,verbose=0)    
    val_mse, val_mae=model.evaluate(val_data,val_targets, verbose=0)
    all_scores.append(val_mae)
    
    
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets=train_targets[i*num_val_samples:(i+1)*num_val_samples]
    
    partial_train_data=np.concatenate(
        [train_data[:i*num_val_samples],
         train_data[(i+1)*num_val_samples:]],
        axis=0)
    
    partial_train_targets=np.concatenate(
        [train_targets[:i*num_val_samples],
         train_targets[(i+1)*num_val_samples:]],
        axis=0)

    model = build_model()
    
    history=model.fit(partial_train_data,partial_train_targets,
                      validation_data=(val_data,val_targets),
                      epochs=num_epochs,batch_size=16,verbose=0)
    
    mae_history=history.history["val_mae"]
    all_mae_histories.append(mae_history)
    
average_mae_history = [
np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()
    
    
all_mae_histories[0] 
len(all_mae_histories[0] )    



import seaborn as sns

sns.histplot(all_mae_histories[3])

max(all_mae_histories[3])
    
truncated_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

model = build_model()
model.fit(train_data, train_targets,
epochs=130, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

predictions=model.predict(test_data)

pred_list=[i[0] for i in predictions ]
len(pred_list)
test_targets

pred_list=np.array(pred_list)

diff=test_targets-pred_list

sns.histplot(diff)

test_targets[np.argmin(diff)]
pred_list[np.argmin(diff)]

plt.scatter(test_targets, pred_list)

np.sqrt(sum(diff**2))/len(diff)

diff2=np.delete(diff,97)

np.sqrt(sum(diff2**2))/len(diff2)
    
    
    