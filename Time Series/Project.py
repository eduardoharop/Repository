# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:23:02 2022

@author: eduar
"""

## Time Series Project
import os 
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

os.getcwd() 
os.chdir('C:/Users/eduar/OneDrive/Escritorio/Masterado/Primer Semestre/Time Series Analysis/Project/AirTemp')

  
# Step 1) Load CSV File into Dataframe
sns.set(rc={'figure.figsize':(15,6)})
df = pd.read_excel('stockholm_daily_mean_temperature2.xlsx')
print(df.info())


# Step 2) Data cleansing
df = df.iloc[52595:97488] ##include data from 1900's
df['date2'] = pd.to_datetime(df['date']) ##create new date
df = df[['date2','adjust']] ##select adjust and new date
df.rename(columns = {'date2':'date'}, inplace = True)
df.set_index('date',inplace=True)
df = df.resample('M').mean()

# Step 3) Plot the data to find patterns
plt.figure(figsize=(15,6))
plt.plot(df.adjust)

# Step 4) Time Series decomposition
decomposition = sm.tsa.seasonal_decompose(df.adjust, model='additive')
plt.rcParams["figure.figsize"] = [16,9]
fig = decomposition.plot()

# Step 5) Check data stationary: ADF unit root test
## Train and Test Datasets
model = df.adjust
train = df.adjust[:'2017-11-30'].to_frame()
test = df.adjust['2017-12-31':].to_frame()


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
check_stationarity(train)

# Plot ACF and PACF of the original series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.figure(figsize=(10,2))
plot_acf(df.adjust, lags=100) ## seasonal patterns decay exponentially
plot_pacf(df.adjust, lags=100) ## seasonal patterns end abruptly 

# Apply seasonal differences and check stationarity
train['adjustdiff'] = train - train.shift(12)
df_diff12 = train['adjustdiff'].to_numpy()
df_diff12 = df_diff12[~np.isnan(df_diff12)] #drop missing data
plt.plot(df_diff12)
check_stationarity(df_diff12)


# Apply both types of differences and check stationarity
train['adjustdiff2'] = train['adjustdiff'] - train['adjustdiff'].shift(1)
df_diff12diff = train['adjustdiff2'].to_numpy()
df_diff12diff = df_diff12diff[~np.isnan(df_diff12diff)] #drop missing data
plt.plot(df_diff12diff)
check_stationarity(df_diff12diff)

# Plot ACF and PACF of seasonal differenced series dfdiff2
plt.figure(figsize=(10,2))
plot_acf(df_diff12, lags=100) ## MA = 3 and SMA = 1
plot_pacf(df_diff12, lags=100) ## AR = 1 and NO SAR



# Plot ACF and PACF of both differenced series dfdiff2
plt.figure(figsize=(10,2))
plot_acf(df_diff12diff, lags=100) 
plot_pacf(df_diff12diff, lags=100)



################################################
# Step 6) Fit SARMA/SARIMA models

# Fit a SARIMA(1,0,3)(0,1,1)12 model
model1 = sm.tsa.statespace.SARIMAX(train.adjust, 
                                order=(1,0,3), 
                                seasonal_order=(0,1,1,12)
                                )
results1 = model1.fit()

print(results1.summary())
# Plot residual errors
resid1 = pd.DataFrame(results1.resid)[12:]
fig, axes = plt.subplots()  
axes.plot(resid1)
axes.set_title("Residuals")

fig, axes = plt.subplots(1, 2, sharex=True)
plot_acf(resid1.dropna(), lags=100, ax=axes[0])
plot_pacf(resid1.dropna(), lags=100, ax=axes[1])
plt.show()
acorr_ljungbox(resid1, lags=[24], return_df=True)


# Fit a SARIMA(1,0,2)(0,0,1)12 model
model2 = sm.tsa.statespace.SARIMAX(train.adjust, 
                                order=(1,0,2), 
                                seasonal_order=(0,1,1,12)
                                )
results2 = model2.fit()
print(results2.summary())
# Plot residual errors
resid2 = pd.DataFrame(results2.resid)[12:]
fig, axes = plt.subplots()  
axes.plot(resid2)
axes.set_title("Residuals")

fig, axes = plt.subplots(1, 2, sharex=True)
plot_acf(resid2, lags=36, ax=axes[0])
plot_pacf(resid2, lags=36, ax=axes[1])
plt.show()
acorr_ljungbox(resid2, lags=[24], return_df=True)


# Fit a SARIMA(1,0,1)(0,0,1)12 model
model3 = sm.tsa.statespace.SARIMAX(train.adjust, 
                                order=(1,0,1), 
                                seasonal_order=(0,1,1,12)
                                )
results3 = model3.fit()
print(results3.summary())
# Plot residual errors
resid3 = pd.DataFrame(results3.resid)[12:]
fig, axes = plt.subplots()  
axes.plot(resid3)
axes.set_title("Residuals")

fig, axes = plt.subplots(1, 2, sharex=True)
plot_acf(resid3.dropna(), lags=36, ax=axes[0])
plot_pacf(resid3.dropna(), lags=36, ax=axes[1])
plt.show()
acorr_ljungbox(resid3, lags=[1], return_df=True)

################################################
# Step 7) Fit Exponential Smoothing models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
hw = ExponentialSmoothing(train.adjust,
                            trend='add',
                            seasonal='add',
                            seasonal_periods=12).fit()
hw_predictions = hw.forecast(60).rename('Holt Forecast')
print(hw_predictions)

train.adjust['2017-1-31':].plot(legend=True,label='Train')
test.plot(legend=True,label='Test',figsize=(12,8))
hw_predictions.plot(legend=True,label='Forecast')


# Step 8) Predict the last 60 observations using all 4 models

df['forecast1'] = results1.predict(start='2017-12-31', end='2022-11-30', dynamic= True) 
df['forecast2'] = results2.predict(start='2017-12-31', end='2022-11-30', dynamic= True) 
df['forecast3'] = results3.predict(start='2017-12-31', end='2022-11-30', dynamic= True) 
df['forecast4'] = hw_predictions

train.adjust['2017-1-31':].plot(legend=True,label='Train')
test.plot(legend=True,label='Test')
results1.predict(start='2017-12-31', end='2022-11-30', dynamic= True).plot(legend=True,label='Forecast')

df[['adjust', 'forecast1']]['2016-11-30':].plot(figsize=(12, 8)) 
df[['adjust', 'forecast2']]['2016-11-30':].plot(figsize=(12, 8)) 
df[['adjust', 'forecast3']]['2016-11-30':].plot(figsize=(12, 8)) 
df[['adjust', 'forecast4']]['2016-11-30':].plot(figsize=(12, 8)) 


df[['adjust', 'forecast1','forecast2','forecast3','forecast4']]['2016-11-30':].plot(figsize=(12, 8)) 

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error 
from sklearn.metrics import mean_absolute_error


test = df.loc['2017-12-31':]

# Forecast Accuracy Measures


forecastmat = pd.DataFrame()  
index = ['forecast1','forecast2','forecast3','forecast4']

rmse = [round(np.sqrt(mean_squared_error(test['adjust'],test['forecast1'])),2),
        round(np.sqrt(mean_squared_error(test['adjust'],test['forecast2'])),2),
        round(np.sqrt(mean_squared_error(test['adjust'],test['forecast3'])),2),
        round(np.sqrt(mean_squared_error(test['adjust'],test['forecast4'])),2)]

mae = [round(mean_absolute_error(test['adjust'],test['forecast1']),2),
       round(mean_absolute_error(test['adjust'],test['forecast2']),2),
       round(mean_absolute_error(test['adjust'],test['forecast3']),2),
       round(mean_absolute_error(test['adjust'],test['forecast4']),2),
       ]

mape = [round(100*mean_absolute_percentage_error(test['adjust'],test['forecast1']),2),
        round(100*mean_absolute_percentage_error(test['adjust'],test['forecast2']),2),
        round(100*mean_absolute_percentage_error(test['adjust'],test['forecast3']),2),
        round(100*mean_absolute_percentage_error(test['adjust'],test['forecast4']),2)
        ]

maen = [round(mean_absolute_error(test['adjust'][1:],test['forecast1'][:-1]),2),
        round(mean_absolute_error(test['adjust'][1:],test['forecast2'][:-1]),2),
        round(mean_absolute_error(test['adjust'][1:],test['forecast3'][:-1]),2),
        round(mean_absolute_error(test['adjust'][1:],test['forecast4'][:-1]),2)
        ]

forecastmat['rmse'] = rmse
forecastmat['mae'] = mae
forecastmat['mape'] = mape
forecastmat['maen']= maen

forecastmat = forecastmat.set_index(pd.Index(index))
print(forecastmat)

# Step 9) Predict the next 60 observations using the best model
hwf = ExponentialSmoothing(df['adjust'],
                               trend='add',
                               seasonal='add',
                               seasonal_periods=12).fit()
forecasts = hwf.forecast(60).rename('predictions').to_frame()
print(forecasts)


forecasts2 = pd.DataFrame({'predictions': 5.12069 },index = ['2022-11-30 00:00:00'])
forecasts2 = forecasts2.append(forecasts)
forecasts2.index = pd.to_datetime(forecasts2.index)

df['adjust']['2017-11-30':].plot(legend=True,label='Passengers',figsize=(12,8))
forecasts2['predictions'].plot(legend=True,label='Holt-Winters Forecast')

