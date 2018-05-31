# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:33:52 2018

@author: jaydeep thik
"""

import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers, losses, metrics
from keras import regularizers
import matplotlib.pyplot as plt

#simple function to vectorize the input data
def vectorize_seq(sequences, dimensions=10000):
    result = np.zeros((len(sequences), dimensions))
    
    for i, sequence in enumerate(sequences):
        result[i, sequence]=1
    return result

#load imdb dataset
(X_train, y_train),(X_test, y_test) = imdb.load_data(num_words=10000)

X_train = vectorize_seq(X_train)
X_test = vectorize_seq(X_test)

y_train =  np.asarray(y_train).astype('float32')
y_test =  np.asarray(y_test).astype('float32')

#a shallow network
network = models.Sequential()
network.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01) , input_shape=(10000,)))
#network.add(layers.Dropout(0.5))
#network.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
network.add(layers.Dropout(0.25))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dropout(0.50))
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(optimizer=optimizers.adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

#generate validation and prtial training set
X_val = X_train[:10000]
X_part = X_train[10000:]

y_val = y_train[:10000]
y_part = y_train[10000:]

#train the model
history  = network.fit(X_part, y_part, epochs=40, batch_size=512, validation_data=(X_val,y_val))

history_dict = history.history
train_acc = history_dict['acc']
validation_acc = history_dict['val_acc']
train_loss = history_dict['loss']
validation_loss = history_dict['val_loss']

eopch = range(1, len(train_acc)+1)
plt.plot(eopch, train_loss, 'bo', label = 'training loss')
plt.plot(eopch, validation_loss, 'b', label = 'validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()


plt.plot(eopch, train_acc, 'bo', label= 'training accuracy')
plt.plot(eopch, validation_acc, 'b', label= 'validation accuracy')
plt.xlabel('eopch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

print("training accuracy: ",network.evaluate(X_train, y_train)[1])
print("testing accuracy: ", network.evaluate(X_test, y_test)[1])
