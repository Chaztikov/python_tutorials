import sys, os, re, pickle, subprocess
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
import datetime

import scipy.integrate
from scipy.spatial import KDTree
from scipy.interpolate import BSpline
from scipy.interpolate import splrep, splder, sproot,make_interp_spline
import scipy.sparse.linalg as spla

style.use('fivethirtyeight')

symbols = 'INTC'
# symbols = 'TSLA' #['TSLA','INTC']
server = 'yahoo'

set_window=7
set_min_periods=0
year0,month0,day0 = 2019,6,6
yearf,monthf,dayf = 2019,6,11
start=datetime.datetime(year0,month0,day0)
end=datetime.datetime(yearf,monthf,dayf)



df = web.DataReader(symbols,server,start,end);
# dframe = pd.concat(dframes,axis=0)

# print(dframe)
# df.to_csv('TSLA.csv')
# df = pd.read_csv('TSLA.csv', parse_dates=True, index_col=0)
df['100ma'] = df['Adj Close'].rolling(window=set_window,min_periods=set_min_periods).mean()


plt.figure(figsize=(12,8))
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax1)
ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax1.plot(df.index, df['High'])
ax1.plot(df.index, df['Low'])
ax2.bar(df.index, df['Volume'])
plt.tight_layout(h_pad=1.04)
plt.show()


# for symbol in symbols:
#     dframe = df.get_cols(symbol)
# with dframe as df:
#     ax1.plot(df.index, df['Adj Close'])
#     ax1.plot(df.index, df['100ma'])
#     ax1.plot(df.index, df['High'])
#     ax1.plot(df.index, df['Low'])
#     ax2.bar(df.index, df['Volume'])
#     plt.tight_layout(h_pad=1.04)
#     plt.show()
# plt.figure()
# df['100ma'].plot()
# df['Adj Close'].plot() 
# df['High'].plot() 
# df['Low'].plot() 
# plt.legend()
# plt.show();



# If you want to know more about subplot2grid, check out this subplots with Matplotlib tutorial.
# Basically, we're saying we want to create two subplots, and both subplots are going to act like they're on a 6x1 grid, where we have 6 rows and 1 column. The first subplot starts at (0,0) on that grid, spans 5 rows, and spans 1 column. The next axis is also on a 6x1 grid, but it starts at (5,0), spans 1 row, and 1 column. The 2nd axis also has the sharex=ax1, which means that ax2 will always align its x axis with whatever ax1's is, and visa-versa. Now we just make our plots:

# print(df.head())
# print(df.columns)
# print(df[['High','Low']])


# import datetime as dt
# import matplotlib.pyplot as plt
# from matplotlib import style
# import pandas as pd
# import pandas_datareader.data as web
# style.use('ggplot')

# df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)
# df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()
# print(df.head())

# ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
# ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

# ax1.plot(df.index, df['Adj Close'])
# ax1.plot(df.index, df['100ma'])
# ax2.bar(df.index, df['Volume'])

# plt.show()