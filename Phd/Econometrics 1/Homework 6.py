# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:57:51 2024

@author: eduar
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

seed =  24228791
np.random.seed(seed)

n = 100
x = np.random.uniform(0, 10, n)
epsilon = np.random.normal(0, 1, n)
y = 4 + 0.8 * x + epsilon

############################################################################
#excercise 1
x_ols = sm.add_constant(x)
model = sm.OLS(y, x_ols).fit()
print(model.summary())

# Monte Carlo Simulation
n_replications = 1000
beta1_vals = []
r_squared_vals = []
data = pd.DataFrame({'x': x, 'y': y})


for _ in range(n_replications):
    bootstrap_sample = data.sample(n, replace=True)
    X_bootstrap = sm.add_constant(bootstrap_sample['x'])
    y_bootstrap = bootstrap_sample['y']
    model_bootstrap = sm.OLS(y_bootstrap, X_bootstrap).fit()
    beta1_vals.append(model_bootstrap.params[1])
    r_squared_vals.append(model_bootstrap.rsquared)
    
# Summary statistics for beta_1 and R^2
beta1_vals = np.array(beta1_vals)
r_squared_vals = np.array(r_squared_vals)

print("Beta_1 mean:", beta1_vals.mean())
print("Beta_1 std deviation:", beta1_vals.std())
print("Beta_1 min:", beta1_vals.min())
print("Beta_1 max:", beta1_vals.max())

print("R^2 mean:", r_squared_vals.mean())
print("R^2 std deviation:", r_squared_vals.std())
print("R^2 min:", r_squared_vals.min())
print("R^2 max:", r_squared_vals.max())

# Kernel Density Plots
sns.kdeplot(beta1_vals, label="Beta_1 Distribution")
plt.title("Kernel Density of Beta_1 Estimates")
plt.show()

sns.kdeplot(r_squared_vals, label="R^2 Distribution")
plt.title("Kernel Density of R^2")
plt.show()

############################################################################
#excercise 2

# Part 2b: Intercept and Slope Using Extremes with Bootstrap
beta0_extremes = []
beta1_extremes = []

for _ in range(n_replications):
    # Bootstrap sample with replacement
    bootstrap_sample = data.sample(n, replace=True)
    x_bootstrap = bootstrap_sample['x']
    y_bootstrap = bootstrap_sample['y']
    min_idx, max_idx = np.argmin(x_bootstrap), np.argmax(x_bootstrap)
    beta1_extreme = (y_bootstrap.iloc[max_idx] - y_bootstrap.iloc[min_idx]) / (x_bootstrap.iloc[max_idx] - x_bootstrap.iloc[min_idx])
    beta0_extreme = y_bootstrap.iloc[min_idx] - beta1_extreme * x_bootstrap.iloc[min_idx]
    beta1_extremes.append(beta1_extreme)
    beta0_extremes.append(beta0_extreme)

# Summary statistics for extreme-based Beta_0 and Beta_1
beta0_extremes = np.array(beta0_extremes)
beta1_extremes = np.array(beta1_extremes)

print("Extreme Beta_1 mean:", beta1_extremes.mean())
print("Extreme Beta_1 std deviation:", beta1_extremes.std())
print("Extreme Beta_1 min:", beta1_extremes.min())
print("Extreme Beta_1 max:", beta1_extremes.max())

sns.kdeplot(beta1_extremes, label="Beta_1 Extreme Distribution (Bootstrap)")
plt.title("Kernel Density of Beta_1 (Extremes, Bootstrap)")
plt.show()

############################################################################
#excercise 3

print("Standard Deviation Comparison:")
print("Bootstrap OLS Beta_1 std deviation:", beta1_vals.std())
print("Extreme-Based Beta_1 std deviation (Bootstrap):", beta1_extremes.std())
print("The standard deviation form Beta_1 from OLS is less than from the Extreme-Based Beta_1, however, since the latter estimator is most likely unbiased because it is based on extreme points, then this situation does not violate the Gauss Markov Theorem because the estimators are not comparable")

############################################################################
#excercise 4

# Part 2d: OLS Without Intercept with Bootstrap
gamma_vals = []
pseudo_r_squared_vals = []
corr_squared_vals = []

for _ in range(n_replications):
    # Bootstrap sample with replacement
    bootstrap_sample = data.sample(n, replace=True)
    x_bootstrap = bootstrap_sample['x']
    y_bootstrap = bootstrap_sample['y']
    model_bootstrap = sm.OLS(y_bootstrap, x_bootstrap).fit()
    gamma_vals.append(model_bootstrap.params[0])
    y_pred = model_bootstrap.predict(x_bootstrap)
    ess = np.sum((y_pred - y_bootstrap.mean())**2)
    tss = np.sum((y_bootstrap - y_bootstrap.mean())**2)
    pseudo_r_squared = ess / tss
    pseudo_r_squared_vals.append(pseudo_r_squared)
    corr_squared_vals.append(np.corrcoef(y_bootstrap, y_pred)[0, 1]**2)

# Summary statistics
print("Gamma mean:", np.mean(gamma_vals))
print("Gamma std deviation:", np.std(gamma_vals))
print("Pseudo R^2 mean:", np.mean(pseudo_r_squared_vals))
print("Pseudo R^2 std deviation:", np.std(pseudo_r_squared_vals))

# Kernel Density Plots
sns.kdeplot(gamma_vals, label="Gamma Distribution (Bootstrap)")
plt.title("Kernel Density of Gamma Estimates (Bootstrap)")
plt.show()

sns.kdeplot(pseudo_r_squared_vals, label="Pseudo R^2 Distribution (Bootstrap)")
plt.title("Kernel Density of Pseudo R^2 (Bootstrap)")
plt.show()

############################################################################
#excercise 5
print("Comparison for Gamma Standard Deviations:")
print("Gamma std deviation (Bootstrap):", np.std(gamma_vals))
print("Bootstrap OLS Beta_1 std deviation from Part 2a:", beta1_vals.std())
print("This situation does not violate the Gauss Markov Theorem because the estimators are not comparable since Gamma is a biased estimator because the intercept is missing")


