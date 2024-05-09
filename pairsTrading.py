import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="whitegrid")

def generate_data(params):
    mu = params[0]
    sigma = params[1]
    return np.random.normal(mu, sigma)

params = (0,1)
T = 100
A = pd.Series(index=range(T))
A.name = 'A'
for t in range(T):
    A[t] = generate_data(params)

B = pd.Series(index=range(T))
B.name = 'B'
for t in range(T):
    params = (t*0.1, 1)
    B[t] = generate_data(params)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
ax1.plot(A)
ax2.plot(B)
ax1.legend(["Series A"])
ax2.legend(["Series B"])
ax1.set_title('Stationary')
ax2.set_title('Non-Stationary')