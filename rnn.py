#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 03:38:29 2018

@author: frank7
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


