# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:17:10 2023

@author: eduar
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import statsmodels as sm

os.getcwd() #get current working directory
os.chdir("C:/Users/eduar/OneDrive/Escritorio/Masterado/Segundo Semestre/Macroeconometrics/Excercises python") #change current directory and change 

df = pd.read_excel("dataset_inf.xlsx")

df['HICP'].plot()
df['HICPenergy'].plot()
df['output'].plot()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df.HICP, lags=20) 
plot_pacf(df.HICP, lags=20) 

plot_acf(df.output, lags=20) 
plot_pacf(df.output, lags=20) 

from statsmodels.tsa.stattools import adfuller
def check_stationarity(timeseries):    
    result = adfuller(timeseries,autolag='AIC')
    dfoutput = pd.Series(result[0:4], 
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    print('The test statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
check_stationarity(df.HICP)

df['HICPdif'] = df.HICP - df.HICP.shift(1) 
dfHICPdif = df['HICPdif'].to_numpy()
dfHICPdif = dfHICPdif[~np.isnan(dfHICPdif)]
check_stationarity(dfHICPdif)

df['HICPdif2'] = df.HICPdif - df.HICPdif.shift(1) 
dfHICPdif2 = df['HICPdif2'].to_numpy()
dfHICPdif2 = dfHICPdif2[~np.isnan(dfHICPdif2)]
check_stationarity(dfHICPdif2)

check_stationarity(df.HICPenergy)
df['HICPenergydif'] = df.HICPenergy - df.HICPenergy.shift(1) 
dfHICPenergydif = df['HICPenergydif'].to_numpy()
dfHICPenergydif = dfHICPenergydif[~np.isnan(dfHICPenergydif)]
check_stationarity(dfHICPenergydif)


check_stationarity(df.output)
adfuller(df.output, regression="c")

df["lHICP"] = np.log(df['HICP'])
df["lHICPenergy"] = np.log(df['HICPenergy'])
df["loutput"] = np.log(df['output'])

from statsmodels.tsa.api import ARDL
model1 = ARDL(df.HICPdif.dropna(),1,df[['HICPenergydif','outputdif']].dropna(),{"HICPenergydif":1,"outputdif":2})
print(model1.fit().summary())

model2 = ARDL(df.lHICP,1,df[['lHICPenergy','loutput']],{"lHICPenergy":1,"loutput":2})
print(model2.fit().summary())

model3 = ARDL(df.lHICP,2,df[['lHICPenergy','loutput']],{"lHICPenergy":2,"loutput":2})
print(model3.fit().summary())

coin_result = sm.tsa.stattools.coint(df.lHICPenergy[1:],df.HICPdif.dropna(),trend = "c")
print(coin_result)
if coin_result[1] < 0.05:
    print("Cointegration")
else: print("no cointegration")

train = df.iloc[0:239]