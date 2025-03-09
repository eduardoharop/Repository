# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:13:10 2025

@author: eduar
"""
import pandas as pd
import os 
import pydynpd as dpd
from pydynpd import regression



os.chdir('C:/Users/eduar/OneDrive/Escritorio/Ph.D/First Year/Spring 2025/Econometrics2/Homework/Homework 2')

data = pd.read_stata("Invest1993.dta")
data = data[["cusip","year","inva","vala","debta","cfa"]]

command_str1='inva L1.inva L2.inva  | gmm(inva, 2:6) | timedumm'
model1 = regression.abond(command_str1, data, ['cusip', 'year'])
print(model1)

command_str2='inva L1.inva L2.inva L1.vala L2.vala L1.debta L2.debta L1.cfa L2.cfa | gmm(inva, 2:6) gmm(vala, 2:6) gmm(debta, 2:6) gmm(cfa, 2:6) | timedumm'
model2 = regression.abond(command_str2, data, ['cusip', 'year'])
print(model2)

model1.models[0].regression_table
model2.models[0].regression_table
