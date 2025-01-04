#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:40:51 2017

@author: chris
"""
#%%
import pandas 
import numpy as np
import matplotlib.pyplot as plt



dataCSV = pandas.read_csv(
        "~/Downloads/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv",
        verbose=False, ) 

data = dataCSV['Weighted_Price']
data = np.asarray(data)


np.sum(np.isnan(data))

plt.figure(1); plt.clf()
plt.plot(data)