import sys, os, re, pickle, subprocess
import bs4 as bs
import requests
print(sys.version)

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.serif']='Times'
mpl.rcParams['font.size']= 6.0
# mpl.rcParams['grid.alpha']= 1.0
# mpl.rcParams['grid.color']= '#b0b0b0'
# mpl.rcParams['grid.linestyle']= '-'
mpl.rcParams['grid.linewidth']= 1.8
mpl.rcParams['lines.linewidth']= 2.5
mpl.rcParams['lines.marker']= 'None'
mpl.rcParams['figure.dpi']= 100.0

import numpy as np
import pandas as pd

import os;import numpy as np;import pandas as pd
import os,sys,re,subprocess
import numpy as np
import scipy
import seaborn as sns

import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import style
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates

import datetime

import scipy.integrate
from scipy.spatial import KDTree
from scipy.interpolate import BSpline
from scipy.interpolate import splrep, splder, sproot,make_interp_spline
import scipy.sparse.linalg as spla

style.use('fivethirtyeight')

symbols = 'TSLA'
# symbols = 'TSLA' #['TSLA','INTC']
server = 'yahoo'

set_window=7
set_min_periods=0
year0,month0,day0 = 2015,6,6
yearf,monthf,dayf = 2019,6,11
start=datetime.datetime(year0,month0,day0)
end=datetime.datetime(yearf,monthf,dayf)



df = web.DataReader(symbols,server,start,end);
# dframe = pd.concat(dframes,axis=0)

# print(dframe)
# df.to_csv('TSLA.csv')
# df = pd.read_csv('TSLA.csv', parse_dates=True, index_col=0)
df['100ma'] = df['Adj Close'].rolling(window=set_window,min_periods=set_min_periods).mean()





# df = pd.read_csv('TSLA.csv', parse_dates=True, index_col=0)

df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)


plt.figure(figsize=(12,8))
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax1.xaxis_date()
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax1)
ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax1.plot(df.index, df['High'])
ax1.plot(df.index, df['Low'])
ax2.bar(df.index, df['Volume'])
plt.tight_layout(h_pad=1.04)
candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
plt.show()


plt.figure(figsize=(12,8))
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax1.xaxis_date()
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax1)
candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
plt.tight_layout(h_pad=1.04)
plt.show()