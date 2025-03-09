# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:48:23 2024

@author: eduar
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import random
from stargazer.stargazer import Stargazer


# Set seed using the last four digits of your CUNY Emplid

np.random.seed(8791)

def dataset(n):
    # Generate x1 ~ N(0,1)
    np.random.seed(8791)
    x1 = np.random.normal(0, 1, n)
    # Generate eta ~ chi-squared(1) for x2
    np.random.seed(8791)
    eta = np.random.chisquare(1, n)
    x2 = eta + 0.5 * x1
    # Generate epsilon_1
    np.random.seed(8791)
    epsilon_1 = 3*np.random.normal(0, 1, n)
    # Generate dependent variable y_1
    y_1 = 1 + x1 + 0.75 * x2 + epsilon_1
    # Generate epsilon_2
    np.random.seed(8791)
    epsilon_2 = 1.2247 * (np.random.chisquare(3, n) - 3)
    # Generate dependent variable y_2
    y_2 = 1 + x1 + 0.75 * x2 + epsilon_2
    # Create DataFrame to store results
    return pd.DataFrame({'y_1': y_1,'y_2': y_2 ,'x1': x1, 'x2': x2})

data1 = dataset(25)
data2 = dataset(100)
data3 = dataset(500)
data = [data1,data2,data3]
data = dataset(20)

reg1 = sm.OLS(endog=data1['y_1'], exog=sm.add_constant(data1[data1.columns[2:]])).fit()
reg2 = sm.OLS(endog=data2['y_1'], exog=sm.add_constant(data2[data2.columns[2:]])).fit()
reg3 = sm.OLS(endog=data3['y_1'], exog=sm.add_constant(data3[data3.columns[2:]])).fit()
reg1.summary()
reg2.summary()
reg3.summary()
stargazer = Stargazer([reg1, reg2,reg3])
stargazer.render_latex()


reg4 = sm.OLS(endog=data1['y_2'], exog=sm.add_constant(data1[data1.columns[2:]])).fit()
reg5 = sm.OLS(endog=data2['y_2'], exog=sm.add_constant(data2[data2.columns[2:]])).fit()
reg6 = sm.OLS(endog=data3['y_2'], exog=sm.add_constant(data3[data3.columns[2:]])).fit()
stargazer2 = Stargazer([reg4, reg5,reg6])
stargazer2.render_latex()

    
np.random.seed(100)


def bootstrap_standard_errors(data, n_bootstrap=1000):
    se_model1_beta1 = []
    se_model1_beta2 = []
    se_model2_beta1 = []
    se_model2_beta2 = []
    np.random.seed(100)
    for _ in range(n_bootstrap):
        # Resample with replacement 
        sample = data.sample(frac=1, replace=True)
        X = sm.add_constant(sample[['x1', 'x2']])
        y_1 = sample['y_1']
        y_2 = sample['y_2']
        # Fit OLS model and get standard error of beta1
        model1 = sm.OLS(y_1, X).fit()
        model2 = sm.OLS(y_2, X).fit()
        se_model1_beta1.append(model1.bse['x1'])  # Retrieve the standard error of beta1 (X1) of y1
        se_model1_beta2.append(model1.bse['x2'])  # Retrieve the standard error of beta2 (X2) of y1
        se_model2_beta1.append(model2.bse['x1'])  # Retrieve the standard error of beta1 (X1) of y2
        se_model2_beta2.append(model2.bse['x2'])  # Retrieve the standard error of beta2 (X2) of y2
    return pd.DataFrame({'y1_se_b1': se_model1_beta1,'y1_se_b2': se_model1_beta2,'y2_se_b1': se_model2_beta1,'y2_se_b2': se_model2_beta2})

np.random.seed(100)
se_25=bootstrap_standard_errors(data1,1000)
np.random.seed(100)
se_100=bootstrap_standard_errors(data2,1000)
np.random.seed(100)
se_500=bootstrap_standard_errors(data3,1000)

# Coefficients for regression between y1 and x1,x2
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes[0, 0].hist(se_25["y1_se_b1"], bins=20, alpha=0.5, label='Normal', edgecolor='k')
axes[0, 0].set_title(f"SE of Beta1 for n = {25} for y1")
axes[0, 0].set_xlabel("Bootstrap SE of beta1")
axes[0, 0].set_ylabel("Frequency")

axes[0, 1].hist(se_25["y1_se_b2"], bins=20, alpha=0.5, label='Normal', edgecolor='k')
axes[0, 1].set_title(f"SE of Beta2 for n = {25} for y1")
axes[0, 1].set_xlabel("Bootstrap SE of beta2")
axes[0, 1].set_ylabel("Frequency")

axes[1, 0].hist(se_100["y1_se_b1"], bins=20, alpha=0.5, label='Normal', edgecolor='k')
axes[1, 0].set_title(f"SE of Beta1 for n = {100} for y1")
axes[1, 0].set_xlabel("Bootstrap SE of beta1")
axes[1, 0].set_ylabel("Frequency")

axes[1, 1].hist(se_100["y1_se_b2"], bins=20, alpha=0.5, label='Normal', edgecolor='k')
axes[1, 1].set_title(f"SE of Beta2 for n = {100} for y1")
axes[1, 1].set_xlabel("Bootstrap SE of beta2")
axes[1, 1].set_ylabel("Frequency")

axes[2, 0].hist(se_500["y1_se_b1"], bins=20, alpha=0.5, label='Normal', edgecolor='k')
axes[2, 0].set_title(f"SE of Beta1 for n = {500} for y1")
axes[2, 0].set_xlabel("Bootstrap SE of beta1")
axes[2, 0].set_ylabel("Frequency")

axes[2, 1].hist(se_500["y1_se_b2"], bins=20, alpha=0.5, label='Normal', edgecolor='k')
axes[2, 1].set_title(f"SE of Beta2 for n = {500} for y1")
axes[2, 1].set_xlabel("Bootstrap SE of beta2")
axes[2, 1].set_ylabel("Frequency")

# Coefficients for regression between y2 and x1,x2
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes[0, 0].hist(se_25["y2_se_b1"], bins=20, alpha=0.5, label='Normal', edgecolor='k')
axes[0, 0].set_title(f"SE of Beta1 for n = {25} for y2")
axes[0, 0].set_xlabel("Bootstrap SE of beta1")
axes[0, 0].set_ylabel("Frequency")

axes[0, 1].hist(se_25["y2_se_b2"], bins=20, alpha=0.5, label='Normal', edgecolor='k')
axes[0, 1].set_title(f"SE of Beta2 for n = {25} for y2")
axes[0, 1].set_xlabel("Bootstrap SE of beta2")
axes[0, 1].set_ylabel("Frequency")

axes[1, 0].hist(se_100["y2_se_b1"], bins=20, alpha=0.5, label='Normal', edgecolor='k')
axes[1, 0].set_title(f"SE of Beta1 for n = {100} for y2")
axes[1, 0].set_xlabel("Bootstrap SE of beta1")
axes[1, 0].set_ylabel("Frequency")

axes[1, 1].hist(se_100["y2_se_b2"], bins=20, alpha=0.5, label='Normal', edgecolor='k')
axes[1, 1].set_title(f"SE of Beta2 for n = {100} for y2")
axes[1, 1].set_xlabel("Bootstrap SE of beta2")
axes[1, 1].set_ylabel("Frequency")

axes[2, 0].hist(se_500["y2_se_b1"], bins=20, alpha=0.5, label='Normal', edgecolor='k')
axes[2, 0].set_title(f"SE of Beta1 for n = {500} for y2")
axes[2, 0].set_xlabel("Bootstrap SE of beta1")
axes[2, 0].set_ylabel("Frequency")

axes[2, 1].hist(se_500["y2_se_b2"], bins=20, alpha=0.5, label='Normal', edgecolor='k')
axes[2, 1].set_title(f"SE of Beta2 for n = {500} for y2")
axes[2, 1].set_xlabel("Bootstrap SE of beta2")
axes[2, 1].set_ylabel("Frequency")
