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






import bs4 as bs
import datetime as dt
import os
import pandas_datareader.data as web
import pickle
import requests


def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers


# save_sp500_tickers()
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'morningstar', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df = df.drop("Symbol", axis=1)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


get_data_from_yahoo()