#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 03:38:29 2018

@author: 
"""
# Recurrent Neural Network

# Part 1.- Data Preprocessing 

#Import the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))

training_set_scaled = sc.fit_transform(training_set)

#Creating a data structure with 60 steps and 1 output
X_train = []
Y_train = []

for i in range(60,1258):
    X_train.append(training_set_scaled[i-60: i, 0])
    Y_train.append(training_set_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshaping 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Part #2 Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM( units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding the second layer and some Dropout regularisation
regressor.add(LSTM( units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the third layer and some Dropout regularisation
regressor.add(LSTM( units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the fourth layer and some Dropout regularisation
regressor.add(LSTM( units=50))
regressor.add(Dropout(0.2))

# Adding the output layer 
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error' )

# Fitting the RNN to the training set
regressor.fit( X_train, Y_train, epochs = 100, batch_size = 32 )









           

