#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:25:44 2021

@author: rantao
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:00:38 2020

@author: rantao
"""

#%%
# Track portfolio peformance

import matplotlib.pyplot as plt
import pandas_datareader as web
from scipy import stats
import seaborn as sns
from datetime import datetime, timedelta

# Performance vintage
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')


# Stocks and weights
tickers = ['NIO','DDOG','DG','FUTU','OXY']
wts = [0.6924367418612271,0.11542569530383111,0.010811325323873109,0.17994680448365322,0.0013794330274155386]

price_data = web.get_data_yahoo(tickers,
                               start = start_date,
                               end = end_date)
# Stock monthly return
ret_data = price_data['Adj Close'].resample('M').ffill().pct_change()
print(ret_data.head())

# Portflio monthly return
port_ret = (ret_data * wts).sum(axis = 1)[1:]
print(port_ret.head())



# Benchmark to S&P 500 etf 
benchmark_price = web.get_data_yahoo('SPY',
                               start = start_date,
                               end = end_date)      

benchmark_ret = benchmark_price['Adj Close'].resample('M').ffill().pct_change()[1:]
print(benchmark_ret.head())

sns.regplot(benchmark_ret.values,
port_ret.values)
plt.xlabel("Benchmark Returns")
plt.ylabel("Portfolio Returns")
plt.title("Portfolio Returns vs Benchmark Returns")
plt.show()


# Beta and Alpha
(beta, alpha) = stats.linregress(benchmark_ret.values,
                port_ret.values)[0:2]
                
print("The portfolio beta is", round(beta, 4))
print("The portfolio alpha is", round(alpha,5))

print(port_ret.mean())
print(port_ret.std())
#%%
