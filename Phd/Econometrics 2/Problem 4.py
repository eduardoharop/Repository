# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:08:32 2025

@author: eduar
"""

import pandas as pd
import numpy as np
import os 
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
import statsmodels.api as sm



os.chdir('C:/Users/eduar/OneDrive/Escritorio/Ph.D/First Year/Spring 2025/Econometrics2/Homework/Homework 1')
# Read the data
data = pd.read_csv("boston.dat", header=None, delim_whitespace=True)
# Assign column names
data.columns = [
"CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
"TAX", "PTRATION", "B", "LSTAT", "MEDV", "latt", "long"
]
# Coordinates
latt = data["latt"].values
long = data["long"].values
mycoords = np.column_stack((latt, long))
# Compute distance matrix in miles (assuming coordinates are in degrees)
mydm = cdist(mycoords, mycoords, metric='euclidean') * 69.0 # Approx conversion of degrees to mil
# Set diagonal elements to 0
np.fill_diagonal(mydm, 0)
# Set distances > 50 miles to 0
mydm[mydm > 50] = 0
# Invert distances
mydm = np.where(mydm != 0, 1 / mydm, mydm)
# Convert to a spatial weights matrix
Wn = mydm / mydm.sum(axis=1, keepdims=True) # Row-normalized weights
# Cleanup
del mydm
data["INT"] = 1
n = data.shape[0]
Y = np.log(data["MEDV"].values).reshape(-1, 1)
X = data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATION', 'B', 'LSTAT']].values
X = sm.add_constant(X)
names = [['CONSTANT','CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATION', 'B', 'LSTAT']]
def concentrated_loglik(lambda_val):
    I = np.eye(n)
    A = I - lambda_val * Wn
    try:
        AY = A @ Y
    except np.linalg.LinAlgError:
        return -np.inf
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return -np.inf
    beta = XtX_inv @ X.T @ AY
    e = AY - X @ beta
    sigma_sq = (e.T @ e) / n
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        return -np.inf
    ll = - (n / 2) * np.log(e.T @ e) + logdet
    return ll

# Optimize lambda
result = minimize_scalar(
    lambda x: -concentrated_loglik(x),
    bounds=(-1, 1),
    method='bounded'
)
lambda_hat = result.x

# Compute beta and sigma squared
I = np.eye(n)
A = I - lambda_hat * Wn
AY = A @ Y
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
beta_hat = XtX_inv @ X.T @ AY
e = AY - X @ beta_hat
sigma_sq_hat = (e.T @ e).item() / n

# Standard errors
cov_beta = sigma_sq_hat * XtX_inv
se_beta = np.sqrt(np.diag(cov_beta))

# For sigma squared
se_sigma_sq = np.sqrt(2 * sigma_sq_hat**2 / n)

# For lambda
A_inv = np.linalg.inv(I - lambda_hat * Wn)
WA_inv = Wn @ A_inv
term1 = np.trace(WA_inv @ WA_inv)
term2 = ( (WA_inv @ X @ beta_hat).T @ (WA_inv @ X @ beta_hat) ).item() / sigma_sq_hat
I_lambda_lambda = term1 + term2
se_lambda = 1 / np.sqrt(I_lambda_lambda)

lambda_hat = lambda_hat.item() 
se_lambda = se_lambda.item() 
# Print results
print("Parameter Estimates:")
print(f"Lambda: {lambda_hat:.4f} (SE: {se_lambda:.4f})")
print("Beta Estimates:")
for i, (names,coef, se) in enumerate(zip(names,beta_hat.flatten(), se_beta)):
    print(f"{names}: {coef:.4f} (SE: {se:.4f})")
print(f"Sigma²: {sigma_sq_hat:.4f} (SE: {se_sigma_sq:.4f})")
