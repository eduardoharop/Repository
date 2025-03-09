# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:56:24 2024

@author: eduar
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

# Define the power function in terms of the t-distribution CDF
def power_function(mu, mu_0, sigma, n, alpha):
    # Critical value from the two-sided t-distribution
    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
    
    # Compute the non-centrality parameters
    non_centrality_lower = (mu_0 - mu) / (sigma / np.sqrt(n)) - t_critical
    non_centrality_upper = (mu_0 - mu) / (sigma / np.sqrt(n)) + t_critical
    
    # Power is the sum of probabilities from both tails
    power = stats.t.cdf(non_centrality_lower, df=n - 1) + (1 - stats.t.cdf(non_centrality_upper, df=n - 1))
    
    return power

# Parameters
mu_0 = 30  # Hypothesized mean
sigma_sq = 10  # Variance
sigma = np.sqrt(sigma_sq)
n = 50  # Sample size
alpha = 0.05  # Significance level

# Range of true means to compute power for
mu_values = pd.Series(np.arange(15, 46, 1))  # True means in the range [15, 45]
power_values = pd.Series([power_function(mu, mu_0, sigma, n, alpha) for mu in mu_values])
table = pd.DataFrame({"mu":mu_values,
                      "power":power_values})


mu_values_2 = pd.Series(np.linspace(15, 45,100))  # True means in the range [15, 45]
power_values_2 = pd.Series([power_function(mu, mu_0, sigma, n, alpha) for mu in mu_values_2])
table_2 = pd.DataFrame({"mu":mu_values_2,
                      "power":power_values_2})


# Plotting the power function
plt.figure(figsize=(10, 6))
plt.plot(mu_values_2, power_values_2, label='Power Function', color='b', linewidth=2)
plt.axvline(mu_0, color='r', linestyle='--', label='Null Hypothesis Mean (mu_0 = 30)')
plt.grid(True)
plt.title('Power Function for the Two-Sided t-Test (CDF-based)', fontsize=14)
plt.xlabel('True Mean (μ)', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
