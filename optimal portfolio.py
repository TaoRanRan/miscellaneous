#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 18:15:20 2021

@author: rantao
"""

# Portfolio optimization

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import seaborn as sns
from datetime import datetime, timedelta

#%%
# Performance vintage
# start_date = (datetime.today() - timedelta(days=500)).strftime('%Y-%m-%d')
# end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=865)).strftime('%Y-%m-%d')
end_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

print(f'start date: {start_date}, end date : {end_date}')

#%%
# Stocks
tickers = ['NIO','DDOG','DG','FUTU','OXY']
price_data = web.get_data_yahoo(tickers,
                           start = start_date,
                           end = end_date)['Adj Close']

log_ret = np.log(price_data/price_data.shift(1))

log_ret.head()
#%%
# Covariance matrix
cov_mat = log_ret.cov() * 252
print(cov_mat)

#%%
cor = log_ret.corr()
sns.heatmap(cor, annot=True,cmap=plt.cm.PuBu)

#%%
# We find the frontier as shown below and either maximize the Expected Returns for Risk level or minimize Risk for a given Expected Return level.
#%%
# Simulating 5000 portfolios
# Trial and error
num_port = 5000
# Creating an empty array to store portfolio weights
all_wts = np.zeros((num_port, len(price_data.columns)))
# Creating an empty array to store portfolio returns
port_returns = np.zeros((num_port))
# Creating an empty array to store portfolio risks
port_risk = np.zeros((num_port))
# Creating an empty array to store portfolio sharpe ratio
sharpe_ratio = np.zeros((num_port))

for i in range(num_port):
  wts = np.random.uniform(size = len(price_data.columns))
  wts = wts/np.sum(wts)
  
  # saving weights in the array  
  all_wts[i,:] = wts
  
  # Portfolio Returns
  port_ret = np.sum(log_ret.mean() * wts)
  port_ret = (port_ret + 1) ** 252 - 1
  
  # Saving Portfolio returns 
  port_returns[i] = port_ret
  
  # Portfolio Risk
  port_sd = np.sqrt(np.dot(wts.T, np.dot(cov_mat, wts)))
  port_risk[i] = port_sd
  
  # Portfolio Sharpe Ratio
  # Assuming 0% Risk Free Rate
  
  sr = port_ret / port_sd
  sharpe_ratio[i] = sr
 
#%%

fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.set_xlabel('Std')
ax1.set_ylabel("Return")
ax1.set_title("Efficient Frontier")
plt.scatter(port_risk, port_returns)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show();


#%%
# Minimum variance portfolio  
names = price_data.columns
min_var = all_wts[port_risk.argmin()]

print("The min variance", round(port_risk.min(),5)) 
dict(zip(tickers, min_var.tolist()))

#%%
# The Sharpe-ratio is the average return earned in excess of the risk-free rate per unit of volatility or total risk.
# Max sharpe ratio portfolio
max_sr = all_wts[sharpe_ratio.argmax()]
print("The max sharpe ratio", round(sharpe_ratio.max(),2))
dict(zip(tickers, max_sr.tolist()))

#%%
